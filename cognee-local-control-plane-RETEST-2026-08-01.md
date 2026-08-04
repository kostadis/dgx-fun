# Cognee local control-plane — retest (2026-08-01)

**One line:** Investigated the "matches open issue #3708, Kuzu orphaned-lock" claim from the [2026-07-04 session](cognee-local-control-plane-2026-07-04.md), found it was based on a misidentified issue, then retested the exact same ToEE corpus against current cognee (well past v1.2.2) — **it completed cleanly in ~27 minutes**, zero errors. The persist-phase hang does not reproduce on current code.

---

## What was wrong in the original doc

"Matches open issue #3708, Kuzu orphaned-lock" was an in-the-moment inference, never checked against the actual issue. It doesn't hold up:

- **#3708** ("Kuzu worker crash leaves orphaned database lock, blocking all subsequent recall() calls") is a real, closed GitHub issue — filed by a *different* reporter (`bchsjdss`), not from this session. The PR that closed it (**#3720**, merged into `dev` 2026-07-05) fixed a *different* race (recreating an engine while a previous worker for the same path was still gracefully shutting down). The literal "reclaim a lock left by a dead/crashed worker" fix does **not** exist on current `main` — two contributors have it open and unmerged right now (**PR #4033**, **PR #4249**, both `REVIEW_REQUIRED`, neither merged as of this writing).
- Kostadis's own filed issue on this topic is **#3870** (the chat-layer robustness writeup — unbounded fan-out, retry stacking, no durability), not #3708. Still open; `Vasilije1990` (maintainer) flagged it priority on 2026-07-17.

## What was tested against current HEAD (2026-08-01, this repo pulled to `e61bbb79c`)

1. **Differential concurrency probe** (10 concurrent `add_nodes()` calls, 4000 DataPoints each, no LLM) against both `cognee==1.2.2` and current HEAD. Neither hung. Neither showed a real concurrency speedup either — turns out Kuzu operations are serialized on the worker's event loop *by design* (documented directly in `cognee_db_workers/harness.py`: "preserving today's serialization for adapters whose underlying library is not thread-safe (e.g. Kuzu)"), unchanged between versions. The "concurrent RPC redesign" fixed client-side request dispatch, not query execution — so the original "session-lock serialization" thesis for the hang was weaker than initial source-reading suggested.
2. **#3708's crash reproducer**, adapted to the graph-engine layer directly (SIGKILL the Kuzu worker mid-write, no clean close/evict, then try to reopen against the same db path). Tried twice. **Did not reproduce** — the fresh engine opened and wrote successfully both times, immediately. The OS released Kuzu's advisory file lock promptly on this machine's local filesystem. Doesn't mean the bug is fake (independent reporter, real repro, two maintainer-acknowledged open PRs) — more likely environment/timing-sensitive (the original reporter was on Oracle ARM64).
3. **The real workload**: same 3 grounding docs (`campaign_state.md`, `party.md`, `planning.md`) + 103 NPC dossiers from `~/src/campaigns/toee/docs/` that never completed across ~6 attempts on v1.2.2. Ran against the live DSpark endpoint (both Spark boxes were mid-experiment on `deepseek-ai/DeepSeek-V4-Flash-DSpark` — used as-is rather than reverting to the original Qwen3-Next-80B config, so this isn't a perfectly isolated same-model comparison). `RAISE_INCREMENTAL_LOADING_ERRORS=false`, `self_improvement=False`, no rate limiter (per the validated learnings from the #3870 thread — a rate limiter made things *worse* on this exact workload back in July).

   **Result: `RememberResult(status='completed', items=106, elapsed=1636.3s)`** — ~27 minutes, zero errors, zero retries. All 5 eval queries (player characters, campaign state, main villain, key NPCs, factions) came back correct and well-grounded in the real content (temple war, Hartsch, Alrrem, Belsornig, Lareth the Beautiful — all accurate).

## Caveat

The LLM backend changed (DSpark instead of the original Qwen3-Next-80B MTP-8/seqs8), so this isn't a clean single-variable comparison — some of the improvement could be the backend, not cognee. But the workload, the corpus, and the failure-prone shape (dense docs + many small NPC dossiers, concurrent document fan-out) are identical, and the result is unambiguous: it completed, where the same shape of run never did on v1.2.2 across six attempts spanning multiple config workarounds.

## Verdict

**Superseded: the original doc's "Local Cognee cannot be made to complete these docs from the outside — it needs library code changes" no longer holds as a blanket statement.** Retest before assuming a patch is needed — a lot changed upstream between 1.2.2 (2026-06-26) and current `main` (2026-08-01), including fixes for the actual #3708 lock-race (distinct from what was assumed at the time) and general hardening (e.g. a new backend-aware overload/pacing policy, `cognee/infrastructure/llm/overload_policy.py`, landed since). The narrower, still-open gap — a lock genuinely outliving a *dead* worker (not just a race during recreation) — remains unpatched upstream (PRs #4033/#4249) and untriggered here; worth another look if a future run does hang with that specific `Lock is held by PID <dead>` signature.

Full working notes: `project_kuzu_ladybug_deadlock_investigation.md` in the `cognee` project's Claude memory.
