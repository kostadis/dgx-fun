# Closets as themes, not as indexes

**TL;DR.** Verbatim drawers answer *where does the content I typed live*. Closets answer *what concept is this query touching on*. They are different kinds of answers and should be different tiers of result, not merged into one ranking.

## What each layer is for

**Verbatim drawers**:
- Deterministic. Cosine over the user's exact words.
- No LLM judgment in the path. The user typed those words; they exist.
- Strength: precision. "Find me the file that contains X."

**Closets (LLM-generated summaries and taxonomy labels)**:
- LLM-generated. Each closet is an LLM's claim about what a file is *about*.
- Strength: breadth. "Find me files in the conceptual neighborhood of X."

These are different *kinds* of answers. One is content, the other is opinion-about-content.

## The inversion we made

We merged them into one ranking via the closet rank-boost in `search_within`. A well-ranked closet could outrank a drawer-direct hit. That asked closets to do precision work — "this specific file is the answer" — when their strength is broad coverage.

Two symptoms we observed:

- **Iter 07.** 3B parroted the worked-example sentence into the output for `rpg_retriever.py`. The LLM's bad closet went into the index unreviewed; search treated it as the truth about that file.
- **Full-palace taxonomy.** 15 different files got classified as `external_service` + `black_box`. Their generic sentences collide in embedding space. The "target" file's match got buried at rank 11+ behind 14 false positives the 3B over-eagerly applied.

The boost-not-gate architecture I'd been pleased with stopped bad closets from *blocking* correct retrieval. It didn't stop them from *corrupting the primary ranking* via the boost mechanism. That's the inversion.

## The right shape

Two tiers in every response:

```
{
  primary: [drawer hits, ranked by verbatim cosine ONLY],
  themes:  [closet/taxonomy hits, deduped to one per source file]
}
```

Read `primary` first. If it's the answer, stop. If it isn't, scan `themes` to find which concept I was actually asking about; drill into that file's verbatim drawers from there.

Flow is: broad theme → specific content. Not the other way around.

## Why this honors the LLM-pipeline rule

The global rule (`~/.claude/CLAUDE.md`): LLMs are renderers, not architects. Scope decisions need a human checkpoint.

- `primary` tier has no LLM scope decision in the path → no checkpoint needed.
- `themes` tier has LLM scope decisions, but they're *labeled as such*. The human reading the results is the checkpoint by default — they look at the cluster, decide if it matches the question, drill in. The LLM's judgment is exposed for review instead of merged silently into the answer.

This is the strong form of boost-not-gate. The weak form (what we built) stops LLM errors from *blocking* correct results. The strong form stops them from *contaminating the primary answer at all*.

## What this means for the code

**Keep:**
- The `closet_seed` mechanism — repurpose it to populate the `themes` tier instead of the primary candidate pool.
- The taxonomy rows currently in the palace — they're good `themes` material. Generic labels are the strength, not the weakness, in this role.
- The `recursive_indexer` doc-length cap — orthogonal correctness fix, still right.

**Change:**
- `mempalace/searcher.py:search_within` returns two lists, not one merged list. Drop the closet rank-boost from drawer ranking entirely.
- `mempalace/mcp_server.py:tool_search_hierarchical` propagates both tiers.
- MCP / CLI / UI callers present them as two visually distinct sections.

**Drop:**
- The mental model that closet quality determines retrieval precision. Specificity comes from drawers, full stop.
- Future iterations chasing a "better" single-sentence LLM closet that surfaces specific files. That's the wrong job for a closet.

## The deeper point

A closet is the LLM's *opinion* about a file. Opinions are useful — they help navigate a corpus you don't already know — but they're not authoritative. The right place for an opinion in a UI is a panel labeled "the system thinks this might be relevant," not woven into the answer you trust.

MemPalace's mission is verbatim recall. The `primary` tier honors that mission directly. The `themes` tier is helpful context the user can take or leave. Mixing the two breaks the mission's contract: "we returned what you typed" stops being true the moment LLM scope decisions are merged into ranking.
