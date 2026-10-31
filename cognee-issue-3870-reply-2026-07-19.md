Thanks both — tried both suggestions against the real workload (spark2, MTP-8, seqs 8, the same setup from the original report). Results were mixed, and one part surprised me enough that I think it's worth writing up precisely rather than just "it worked" / "it didn't."

## `RAISE_INCREMENTAL_LOADING_ERRORS=false` — works as described, plus a gotcha

Confirmed in the code first: `remember()` never overrides `cognify()`'s `incremental_loading` default (`True`), so this flag is already live on every `remember()` call today, not something you have to opt into separately (`run_tasks_data_item.py:155`). A fresh single-doc run with it set completed normally with no side effects I could find. This part does what you said — keeping it on.

One thing I don't think is documented anywhere: because a failed data item is never marked `DATA_ITEM_PROCESSING_COMPLETED`, and incremental loading re-scans the whole dataset for incomplete items on every call, **a doc that fails once gets silently re-queued on every subsequent `remember()` call against that dataset** — the cost compounds run over run instead of being skipped or surfaced. On a fixed-capacity backend where a doc can fail from a timeout rather than a real error, that means your Nth retry is doing 1×, 2×, 3×... Nx the LLM work of your 1st, silently. Might be worth a `DATA_ITEM_PROCESSING_FAILED` status (distinct from "never attempted") so a caller can choose to skip vs. retry, rather than it being implicit.

## Rate limiter — made things worse on the real workload, not better

This is the surprising part. `LLM_RATE_LIMIT_REQUESTS=3` / `INTERVAL=60` (chosen via Little's Law against spark2's 8 MTP slots and ~2-4min/call) + `EMBEDDING_RATE_LIMIT_REQUESTS=20/60s`, run against the actual 4-document grounding batch from my report (dense docs, 30min OS timeout each, fresh graph):

**All 4 documents hit the full 30-minute timeout with zero logged progress.** Not slow — zero. `campaign_state.md`'s `extract_graph_and_summarize` produced no "Completed graph extraction" events at all in 30 minutes. That's strictly worse than my original (no-rate-limiter) baseline, which completed extraction in ~10 minutes before wedging in the persist phase. I confirmed vLLM itself was healthy and answering in <1s throughout via direct curl, so this isn't a dead backend.

I isolated it before assuming the limiter was just "too conservative":
- The limiter's own admission logic checks out fine standalone (`hit_limit()` allows exactly 3 hits, then correctly blocks).
- A trivial 1-2-chunk test doc completed in ~1 minute with the limiter on — because a doc that small never even reaches the 3-request ceiling, so the throttle never actually engages. A quick smoke test won't expose this.
- The stall only appears at the real fan-out (~15-30 concurrent chat calls for a 4k-word doc). I didn't fully root-cause it beyond that — candidates I'd look at are a concurrency/scaling issue in the `limits`-library moving-window check under many concurrent coroutines, or an interaction between the LLM and embedding limiters when both are active simultaneously. Happy to help reproduce this on your side if useful — I can share the exact doc/config that triggers it.

Worth noting explicitly for anyone else trying this: `llm_rate_limit_*` is a **moving-window RPM admission throttle**, not a concurrency semaphore — it paces how fast *new* requests are let in, it doesn't cap how many are in flight at once. That distinction matters for point 1 in my original report (unbounded concurrent fan-out) — it's a different lever than what I described there, and per the above it doesn't currently seem to be a safe substitute for one on a workload like this.

## Net effect

`RAISE_INCREMENTAL_LOADING_ERRORS=false`: real win, keeping it.
Rate limiter: reverted to off — net negative as tested.

The persist-phase wedge from my original report (the Kuzu/LanceDB subprocess IPC hang) is untouched by either of these — that's still the harder problem underneath all of this, and neither knob was ever going to reach it.
