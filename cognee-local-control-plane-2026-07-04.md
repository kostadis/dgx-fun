# Cognee on the DGX Spark — the local control-plane session (2026-07-04)

**One line:** Wired Cognee (knowledge-graph memory) to the local Spark instead of a cloud API as a calibration exercise. Got a genuinely useful hardware/throughput result and a sharp architectural thesis; **never got a completed graph** — because a cloud-built tool has no control plane for a fixed-capacity local stack, and the gap is on *every* dependency, not just the LLM.

**Companion docs:** design thesis → `~/cloud-tools-on-local-llms.md`; posted GitHub issue draft → `~/cognee-github-issue.md`; live Spark inventory (spark2 now MTP-8) → `~/src/dgx/current-setup.md`; memory → `reference_cognee_spark_setup.md`, `project_cognee_overnight_digestion.md`.

---

## Setup

- **Cognee 1.2.2**, `LLM_PROVIDER="custom"` → `GenericAPIAdapter` → local **vLLM** (Qwen3-Next-80B-A3B-Instruct-FP8) at an OpenAI-compatible endpoint. Embeddings: **Ollama** `qwen3-embedding:0.6b` via `EMBEDDING_PROVIDER="openai_compatible"`. Config in `~/cognee-spark/.env`.
- **Workload:** ToEE grounding docs (4, ~2–4.8k words each, dense) + 104 NPC dossiers (~231 words each). ~55k words total.
- **Routing gotcha (cost hours):** `import cognee` runs `dotenv.load_dotenv(override=True)`, which **clobbers any `os.environ["LLM_ENDPOINT"]`** you set before the import. The `.env` file is the *only* reliable way to point the endpoint. (An earlier "measure on spark2" run was silently defeated by this and actually hit spark1.)

---

## Part 1 — Hardware / throughput calibration (the real yield)

The Cognee workload is **output-decode-bound**: each graph-extraction call generates a large `KnowledgeGraph` JSON (thousands of tokens), and decode on the GB10 is memory-bandwidth-bound (~273 GB/s). Three configs, measured on the actual load:

| Config | Aggregate | per-stream | Notes |
|---|---|---|---|
| **Latency** — MTP-2 spec decode, seqs 3 | 92 t/s | ~30.7 | high per-stream, only 3 slots |
| **Throughput** — plain, seqs 16 | ~125 t/s | ~7.8 | needs all 16 slots to get there |
| **MTP-8** — MTP-2 spec decode, seqs 8 | **~138 t/s** | ~17.3 | **WINNER** |

**MTP-8 wins** (+10% over plain-16, +50% over MTP-3), and reaches ~138 t/s with **8 slots** where plain needs 16. Why: **MTP draft acceptance on the rigid JSON is 86–93%** (mean accept length ~2.8 of 3) — far above the ~77% on hard code — because structured output is highly predictable, so spec decode drafts it almost perfectly. Each MTP stream decodes ~2.3× a plain stream; 8 fast streams beat 16 bandwidth-contended ones, and the rejected-draft compute waste hasn't bitten by seqs 8.

Clean decomposition (from a matched-batch capture): **MTP ≈ 2.3× per stream** (16.5 vs 7.2 t/s), **batching 3→16 ≈ 3×** aggregate. Both seqs extremes (3 and 16) were on the wrong side of the crossover.

Other findings:
- **APC prefix caching is immaterial** for this workload. The prompt *is* cache-friendly (fixed system prompt + schema first, variable chunk last), but the job is decode-bound — caching a ~1k-token prefix saves <3% of a multi-minute decode. Measured ~0% hit rate. (Left on because it's free; KV even went *up* to 6.41×.) APC's real payoff is prefill/TTFT-bound work, not decode-bound batch.
- **Measurement humility:** an early "60 t/s" reading was a warm-up/prefill-heavy transient; steady state is ~130 t/s. Don't trust the first sample.
- spark2 is now configured **MTP-2 + APC + seqs 8** (see `current-setup.md`). Workload-specific: JSON/structured decode → MTP-8; long-context or high-concurrency batch → plain-16 (higher KV headroom).

---

## Part 2 — How Cognee ingestion works

`remember()` = `add()` (store raw) + `cognify()` (build graph — all the LLM work) + `improve()` (self-improvement loop; **skipped** with `self_improvement=False`).

The cognify pipeline per document: `classify_documents` → `extract_chunks_from_documents` → **`extract_graph_and_summarize`** → `add_data_points` (embed + DB write) → `extract_dlt_fk_edges`.

**The burst structure:** documents are gated by `asyncio.Semaphore(data_per_batch=20)` (continuously refilled — as one doc finishes, the next is admitted). Inside a doc, `extract_graph_and_summarize` fires **2 chat calls per chunk** (graph + summary) via a **bare `asyncio.gather` with no LLM concurrency cap**. Net: ~40–48 concurrent chat requests ("bursts"), and vLLM's `max_num_seqs` is the *only* throttle — cognee has none. The queue is kept continuously full (good for throughput, bad when combined with a cloud-tuned timeout).

**Databases (all embedded, local defaults):**
- **Graph:** "Ladybug" = a wrapper around **Kuzu**, run as a **subprocess** (`LadybugAdapter.create_subprocess`). Files: `cognee_graph_ladybug` + `.wal`.
- **Vector:** LanceDB (`cognee.lancedb/`).
- **Relational/metadata:** SQLite (`cognee_db`, `cache.db`).

---

## Part 3 — The failure modes (the control-plane thesis)

Cognee's resilience model is correct for an elastic, always-up, low-latency **cloud** API and wrong for a fixed-capacity local stack. The assumption is **diffuse** — baked across timeouts, retries, commit strategy, concurrency, and error handling — so every fix exposed the next gap:

1. **Unbounded chat fan-out** — no cap on the per-chunk gather; overcommits a fixed-slot backend 3–15×.
2. **Timeout counts queue-wait → retry storm** — the client timeout runs from *send*, so it includes server-queue time; a request waiting in a deep queue times out *while alive*, cognee retries it, and the retry deepens the same queue.
3. **Layered retry ≈ 6× the timeout** — litellm's internal `num_retries` (~3×) × instructor's `MAX_RETRIES=2`; a 300s timeout became ~1800s of churn per straggler (observed `timeout value=300.0, time taken=901.35 seconds` ×2).
4. **All-or-nothing durability** — one straggler's `InstructorRetryException` propagates through the gather and aborts `remember()`, discarding *the whole run* (50 min of extraction → **empty graph**). No checkpointing.
5. **No liveness signal** — the only health check is a one-shot 30s startup preflight (known to false-fail on local; issue #2752). Non-streaming structured output means the client can't tell "queued/slow" from "hung."
6. **The hang lives on every endpoint** — chat (litellm), embed (Ollama), **and** the graph-persist path. The final wedge was a stuck IPC between cognee and its **Kuzu subprocess**: main worker in `epoll_wait` ↔ graph subprocess workers in `pipe_read`, no progress (matches open issue **#3708**, Kuzu orphaned-lock).

---

## Part 4 — Every workaround tried, and why each failed

| Attempt | Result | Why it failed |
|---|---|---|
| spark1 latency (MTP, seqs 3) overnight | died ~3h, 253 timeouts | 3-slot queue starvation → retry storm |
| spark2 throughput (plain, seqs 16) | mid-run wedge | a single hung request, no liveness |
| **spark2 MTP-8** | ran clean ~50 min, then crashed | slow-motion retry storm: deep-queue tail times out (queue-wait), retries deepen it |
| batched `remember()` of 6 docs | batch 1 = 61 min → failed | the 4 dense grounding docs landed in batch 1; retry churn + volume |
| one-doc-at-a-time + hard OS `timeout 600` | small docs OK; heavy docs killed | a single dense grounding doc needs **>10 min of extraction alone** |
| heavy timeout → 1800s (30 min) | extraction ~10 min, then **wedged** | the wedge moved to the **persist phase** — Kuzu subprocess IPC deadlock; a hard kill just loses the doc, next doc wedges identically |

Key sub-finding: for a dense doc, **extraction and persist are each ~10 minutes** — the DB/embed phase roughly *doubles* per-doc time and is where the fatal wedge lives. A single "straggler" chunk (one dense chunk → ~10k-token graph) decodes alone for ~5 min at the tail (the per-document `gather` barrier).

---

## Verdict & thesis

**Local Cognee cannot be made to complete these docs from the outside — it needs library code changes** (or run the LLM+embed on a hosted API). Concurrency scoping, hard OS timeouts, per-unit isolation, and durable append all help, but none touch the persist-phase hang, because it's in a component (the Kuzu subprocess IPC) with no external knob.

> **Thesis:** *Building against the cloud model API's control plane makes a tool very hard to use on backends that don't share the cloud's properties* — and the gap isn't only the model. It's every dependency the tool assumes is elastic, always-up, and responsive: the LLM, the embedder, and even the embedded database. **The cloud hides the control plane, so tools built on it never grow one; take them off the cloud and there's nothing to fall back on.** The convenience that makes cloud tools easy to build is exactly what makes them unable to live anywhere else.

The practical implication for local AI: owning the box and the weights isn't enough — the *entire tool stack* (memory layers, RAG frameworks, agent orchestrators, their embedded DBs) has to be control-plane-aware, and almost none of it is yet, because it grew up on the cloud's hidden one.

---

## Loose ends for a future session

- **Fix mempalace search:** the `~/.mempalace/palaces/chat` palace has a backend mismatch (`chroma` + `turbovec` artifacts coexist) — every search errors until resolved (pick one backend or use a fresh palace dir).
- If the graph is still wanted: fix cognee's code (the issue lists the concrete changes), or point LLM+embed at a hosted API. Do **not** re-attempt external timeout-tuning — it's proven not to work.
- `~/cognee-spark/` holds the experiment scripts (`remember_unit.py`, `heavy_phase.sh`, `npc_phase.sh`, `orchestrate_phases.sh`, `ingest_grounding.py`) and a likely-orphaned Kuzu lock; a `cognee.prune.prune_system()` / fresh `.cognee_system` is advisable before any retry.
