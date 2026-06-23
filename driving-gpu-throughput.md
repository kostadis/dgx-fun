# Driving GPU Throughput Is Not Just Prompt Engineering

A field guide, written from a real run: converting ~1,300 tabletop-RPG PDFs to
[5etools](https://5e.tools) JSON on two local **DGX Spark** boxes serving
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` under vLLM. No Anthropic API — the whole
point was to drive *local* GPUs hard.

The naive mental model is: "write a good prompt, send it, get tokens back." That
gets you a working demo. It does **not** get you a saturated GPU, and it will
happily crash your machine. Everything below is the part nobody mentions when
they talk about "using an LLM" — the systems layer between your prompt and the
silicon.

> **TL;DR.** The prompt was maybe 10% of the work. The other 90% was input
> sizing, output budgeting against the context window, matching client
> concurrency to server batch slots, keeping a buffer, bounding the *driver's*
> memory, balancing load across boxes, feeding only the data that matters, and
> handling failure. Each of those is a knob, each interacts with the others, and
> getting one wrong wastes the GPU or kills the host.

---

## 0. The setup, so the numbers mean something

- **Model:** `Qwen3-Next-80B-A3B-Instruct-FP8`, served by vLLM 0.22.0.
- **Two boxes** ("spark1" / "spark2"), each `--max-num-seqs 16` (16 concurrent
  generation slots), `--max-model-len 131072` (128K context; 256K native).
- **Throughput shape** (this matters more than any single number):
  - **single-stream:** ~20 tok/s. One request at a time wastes the box.
  - **16 concurrent:** ~150–170 tok/s *aggregate* — but only ~10 tok/s *per
    request*. Concurrency is how you get aggregate throughput; it costs you
    per-request latency.
- **The driver:** `batch_convert.py` walks the PDF tree, and per document shells
  out to a converter (`pdf_to_5etools_v2.py`) that chunks the PDF and streams
  chunks to a box.
- **Monitoring:** `spark-tps.sh` polls vLLM's `/metrics` and prints, per box,
  `tok/s`, `Nr` (`vllm:num_requests_running`), `Nw` (`vllm:num_requests_waiting`
  — the queue). You will live in these three numbers.

Keep that `running` / `waiting` distinction in your head. Almost every lesson
below is read off those two gauges.

---

## 1. Input sizing: chunking is not free, and the context window is a hard wall

**What we hit.** A 55-page module had a *flat* bookmark tree: one bookmark
covered pages 10–55 — the entire adventure — as a single 189,414-character
section. That's ~48,000 input tokens in one prompt. The box's read-timeout is
600 s; generating the corresponding JSON blew past it, three times, before the
doc failed. ~30 minutes of wall-clock burned to produce nothing.

**Why.** Two separate ceilings, easily confused:

- **`max-model-len` (128K here)** bounds `input_tokens + max_tokens` for a single
  request. vLLM *rejects* a request that asks for more. So your input budget and
  your output budget are not independent — they share one context window.
- A converter that "just chunks harder" to fit can silently shred a section into
  fragments the model can't reassemble correctly. Chunking is a *scope decision*,
  not a free knob.

**What we changed.** We made the input cap a **hard error, not a chunk
boundary**: if a chunk's prompt exceeds 40,000 tokens, abort the whole document
(write no JSON, exit non-zero) and surface it for a different approach
(`--force-marker` to re-derive structure, or a bigger-context model). The
oversized-leaf case now fails in ~80 s with a clear message instead of three
600 s timeouts.

**Principle.** *Decide deliberately what happens when input doesn't fit.* The
options are: reject, route elsewhere, or restructure — never "silently make it
fit." And remember the wall: `prompt_cap + output_cap < max-model-len`. We run
`40k + 80k = 120k < 131072`. Push either past the window and every large request
400s.

---

## 2. Output budgeting: `max_tokens` is bounded by the *served context*, not the model's spec sheet

**The question we asked.** "The output cap is 50k — can we make it 100k?"

**The answer is provider-specific, and that's the lesson.** `MAX_OUTPUT_TOKENS`
is just the `max_tokens` you send. Its real ceiling depends entirely on *who's
serving*:

- On the **Anthropic API**, Haiku/Sonnet cap output at 64K — 100k would 400 on
  the spec alone.
- On **vLLM**, there is no separate output cap; the ceiling is
  `max-model-len − input_tokens`. With a 40k input cap and a 128K window, the
  real headroom is ~88k. So `100k` *fits only if the input is small*; a request
  near the 40k input cap asking for 100k output = 140k > 128K → **rejected** —
  and it's exactly the big docs that would trip it.

We set the output cap to **80k** (safe at 128K for any input up to the 40k cap)
and made it env-overridable so the same code stays correct on the API path
(where 50k is right) and the vLLM path (where 80k is right).

**Principle.** "Max output tokens" is not a property of the model — it's a
property of *the deployment*. The same constant is wrong on two different
backends. Compute it from the serving config, don't copy it from a blog post.
(And to estimate tokens without a tokenizer for every provider, we used the
boring `chars / 4` heuristic — good enough for routing decisions.)

---

## 3. Concurrency is *the* throughput lever on a batching server

**The single most important number:** ~20 tok/s single-stream vs ~150–170 tok/s
at 16 concurrent on the *same box*. An 8× difference, and the model and prompt
are identical. The win is entirely in how you *schedule* requests against the
server's batch.

vLLM batches: `--max-num-seqs N` is how many sequences it decodes at once. If you
send one request at a time, N−1 slots sit idle and you get single-stream speed.
To get aggregate throughput you must keep ~N requests in flight.

In our driver, in-flight requests is a product of two knobs:

```
in-flight requests per box  =  pool (concurrent docs)  ×  doc-concurrency (chunks per doc)
```

`pool × doc-concurrency` should target the box's `--max-num-seqs`. With seqs 16,
both `pool 16 × conc 1` and `pool 4 × conc 4` give 16 in-flight — *very*
different in every other respect (see §6), but the same GPU load.

**Principle.** Throughput on a batching engine is a *scheduling* problem. The
prompt determines quality and token count; **concurrency determines whether the
GPU is busy.** Tune the prompt for correctness, tune concurrency for utilization
— they are orthogonal.

---

## 4. Matching client concurrency to server slots: under-feeding, oversubscription, and the buffer

We watched `spark-tps.sh` show this, repeatedly, and learned to read it:

| Reading | Meaning |
|---|---|
| `15r 0w` | **Under-fed.** A slot is idle and nothing is queued to fill it. Lost throughput. |
| `16r 0w` | Exactly saturated, zero margin — any hiccup drops you to 15r. |
| `16r 5w` | **Saturated with a buffer.** All slots busy *and* a few queued, so a freed slot backfills instantly. This is the target. |
| `4r 2w` on a seqs-4 box | Saturated + queue (fine). On a *seqs-16* box, `…2w` would mean over-subscription. |

Why `15r 0w` happens even at "16 concurrency": between requests, a client spends
real time *not sending* — parsing the response, validating it, writing files,
preparing the next request. During that gap its slot empties. With no queue
(`0w`), nothing backfills it, so the box averages 15/16 busy.

**The fix is a buffer: deliberately oversubscribe the client a little** so a few
requests are always queued (`w` small-positive). When a slot frees mid-gap, vLLM
immediately starts a queued request. We moved from `pool 16` (→ `15r 0w`) toward
a slightly higher pool (→ `16r 5w`) and recovered that idle ~6%.

**Principle.** Aim for *saturated-plus-small-queue*, not *exactly-N*. A shallow
queue is not waste — it's the shock absorber that hides client-side and network
gaps. Watch `w`: zero means you're leaving throughput on the table; large-and-
growing means you've over-subscribed and are adding latency.

---

## 5. Heterogeneous and *changing* hardware

Real boxes aren't identical, and they don't stay up.

**Asymmetry.** At one point spark1 ran plain at seqs 16 (throughput) and spark2
ran MTP-2 speculative decoding at seqs 4 (latency). Per request: ~10 tok/s on
spark1 (16-way, no spec) vs ~26 tok/s on spark2 (4-way + MTP). Note spark1 ran
**4× the concurrency for only ~1.5× the aggregate** — a sign it was past its
efficiency knee. A single `--pool` couldn't serve both: at `pool 6` spark1 was
starved (6/16) and spark2 was over-subscribed (6 docs vs 4 slots → a queue). We
added **per-box pool** (`--pool1`/`--pool2`) so each box runs at its own seq
count. The general point: *one global concurrency setting cannot drive boxes with
different batch capacities.*

**Reconfiguration mid-run.** Restarting vLLM to change config takes a box offline
for seconds-to-minutes. What happens to in-flight work?

- **Fast restart (< ~70 s):** the transport's own retry/backoff (10→20→40 s)
  bridges it — the request succeeds on a retry, the doc survives.
- **Slow restart (model reload, minutes):** the doc exhausts its retry window and
  is marked `failed`. There is **no auto-failover** to the healthy box — a worker
  bound to the down box just fails its docs. But the workers don't die; once the
  box is back they resume succeeding. Nothing is lost: failed docs have no output
  file, every chunk completed before the crash is cached, and the manifest
  persists after every doc — so re-running the exact command resumes cleanly.

**Principle.** Throughput engineering includes the *unhappy* path. Know your
retry windows, know whether your dispatcher fails over or fails in place, and
design for "a box vanished" as a normal event, not an exception. Cache partial
work so an outage costs minutes, not a re-run from zero.

---

## 6. The driver's own footprint will kill you before the GPU does

This is the one that actually crashed the machine, and the one most "prompt
engineering" framing completely ignores.

**What we hit.** Chasing throughput, we ran `--pool 20` on each of two boxes. The
driver spawns **one Python subprocess per document**, so that's **40 concurrent
Python interpreters**, each importing PyMuPDF and loading a whole PDF — ~200 MB
apiece, ~8 GB total. WSL ran out of memory and the whole run died. The GPUs were
fine. The *client host* fell over.

**The key realization:** in this driver, *in-flight requests* and *number of
processes* were coupled (`pool` controlled both). But the GPU only needs ~32
in-flight requests (16/box) — and you can produce those with far fewer processes
by raising **chunk-concurrency** instead of process-concurrency:

```
--pool 20 --doc-concurrency 1   → 40 processes, 40 requests   (8 GB, OOM)
--pool 4  --doc-concurrency 4   →  8 processes, 32 requests    (1.6 GB, fine)
```

Same GPU load, one-fifth the host memory — a pure config change, no rewrite.

**The trade-triangle.** Once decoupled, three properties pull against each other
for a fixed request target:

- **Memory** ∝ number of interpreters (≈ `pool`). Fewer = lighter.
- **Crash isolation** comes from process-per-doc (a bad PDF can't take down the
  batch). Fewer, longer-lived processes = less isolation.
- **Feed smoothness** ∝ number of independent docs in flight (§7). Fewer docs
  with bigger per-doc bursts = choppier.

You cannot maximize all three. We chose low memory (8 interpreters) and accepted
choppier feeding and slightly coarser isolation. A "lighter-weight, drive more
load" rewrite (one process, a thread-pool of HTTP senders, bounded PDF loaders)
exists and is ~3–4× lighter still — but it trades away crash isolation and needs
special-casing for the GPU-bound extraction path. That's a real engineering
decision, not a default.

**Principle.** The client is a distributed-systems component with its own
resource budget. Decouple "work in flight on the GPU" from "processes/threads on
the host," and pick your point on the memory ↔ isolation ↔ smoothness triangle
*on purpose*. A saturated GPU is worthless if it OOMs the machine feeding it.

---

## 7. Load balancing and feed smoothness

After the memory fix we saw spark1 drain `16r → 11r` (at `0w`) while spark2 held
`16r 7w`. Same hardware — why?

Two compounding effects, both downstream of "few docs per box, big bursts":

1. **Doc-size variance + the tail.** With `doc-concurrency 4`, a long doc keeps a
   worker busy for minutes while contributing only its *tail* of requests near
   the end (its last 1–3 chunks, not a full 4). If a box catches a cluster of
   heavy docs, several workers sit in low-concurrency tails and the box sags.
2. **No buffer to hide it.** spark2 kept a 7-deep queue that backfilled every
   freed slot. spark1's queue drained to `0w`, so every gap showed up directly as
   a drop. The effect is self-reinforcing: the box that falls behind loses its
   buffer and dips harder.

It self-corrects when the heavy docs finish. The lever is *granularity*: more
docs per box with smaller per-worker bursts (`pool 8 × conc 2` instead of
`pool 4 × conc 4`) averages out size variance and keeps a buffer on both boxes —
at the cost of more interpreters (back to §6's triangle).

**Principle.** A shared work-queue across heterogeneous-sized items doesn't
self-balance when each worker carries a big slice of a box's capacity. Smoothness
comes from *many small independent units of work*, which costs memory. There is
no free lunch; there is a dial.

---

## 8. Don't spend GPU on the wrong data

The cheapest throughput win isn't faster generation — it's **not generating what
you don't need.** The PDF library was full of old versions and format variants of
the same product: `Book.pdf`, `Book_PrintFriendly.pdf`, `Book_(Optimized).pdf`,
`Book_v1.4` / `v2`, plus Pathfinder editions and map booklets. Converting all of
them is pure waste.

We reused an *existing* library database (the `rpg-lib` indexer already computes
`is_old_version` / `is_draft` / `is_duplicate`) and added a thin "pick one
canonical per product" pass (prefer the printer-friendly edition — cleaner text
layer; keep the newest version; skip maps and Pathfinder editions). Result:
**1,075 pending conversions → 747.** A ~30% throughput gain with *zero* change to
the model, prompt, or serving config.

**Principle.** Before optimizing how fast you process each item, ask whether the
item should be processed at all. Deduplication, filtering, and "convert the right
variant" are throughput levers that live entirely *upstream* of the GPU.

---

## 9. Failure handling and observability are part of "throughput"

A run that does 165 tok/s but silently drops 200 docs hasn't done 165 tok/s of
*useful* work. We built a failure taxonomy and made each visible:

- **Oversized-leaf** (§1) → fail fast with the section + page range, re-run with
  `--force-marker`.
- **Endpoint outage** (§5) → bridged or failed-and-resumable, never silent.
- **Flat-bookmark structure** → the converter can't chunk it; surfaced as a
  distinct failure, not a hang.
- The driver persists the manifest after every doc, caches every chunk
  (`--reuse-responses`), and a re-run mops up `failed` docs without re-billing
  completed work.

And we learned to *read the live signal*: `running` vs `waiting` told us under-
feeding from saturation from over-subscription; a monotonic drain on one box told
us load imbalance; a flatline-and-stay-low would tell us a stuck retry chain.

**Principle.** Observability is not optional decoration. On a batching server the
two gauges `running` and `waiting` are your throughput dashboard, and a typed
failure taxonomy with resumable state is what lets a 12-hour unattended run
actually finish.

---

## The point

Here is the full set of knobs we touched. Exactly **one** of them is "the
prompt":

| Knob | Governs | Get it wrong and… |
|---|---|---|
| Prompt / system prompt | Output quality & token count | wrong content (but the GPU's still busy) |
| Input cap (chunk vs error) | Whether a doc fits the window | silent shredding or 600 s timeouts |
| Output `max_tokens` | Output budget **vs context window** | every large request 400s |
| `max-num-seqs` (server) | Batch width | low ceiling on aggregate throughput |
| `pool × doc-concurrency` (client) | In-flight requests | idle GPU, or OOM host |
| Oversubscription / buffer (`w`) | Hiding inter-request gaps | under-fed slots |
| Per-box pool | Driving heterogeneous boxes | starve one, flood the other |
| Process vs thread model | Host memory & isolation | crash the machine feeding the GPU |
| Work granularity | Feed smoothness & balance | one box sags while the other holds |
| Dedup / data selection | *Whether to process at all* | 30%+ of throughput wasted on duplicates |
| Retry / resume / manifest | Surviving outages | lost work, non-resumable runs |
| `running` / `waiting` gauges | Knowing which of the above is wrong | flying blind |

Prompt engineering decides *what the model says*. Everything else on that list
decides *whether the GPU is busy, whether the host survives, and whether the run
finishes.* Driving GPU throughput is a systems-engineering problem with a prompt
attached — not the other way around.
