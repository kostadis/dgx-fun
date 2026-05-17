# D&D session prep with Opus as the orchestrator

**Goal:** Use Opus (frontier model, smart, expensive) to *decide themes
and structure* for the next D&D session, while delegating all
detail-retrieval to mempalace+Gemma (cheap, fast, local). Replace the
"load the whole corpus into the prompt" pattern with "load thin
grounding + agentic retrieval as needed."

**Status:** not built. This is the spec.

**Authors:** Kostadis + Claude.

**Sits inside:** the broader operational framework in
[`tiered-llm-workflow.md`](./tiered-llm-workflow.md). This doc is the
**Tier 1 + Tier 3 hybrid** instantiation for the specific case of D&D
session prep.

## The problem with the current shape

Today, `~/src/CampaignGenerator/prep.py` runs a multi-pass LLM pipeline
(per `GMASSISTANT_PIPELINE.md`) where each pass has a fixed prompt
template. The LLM is asked to produce structured output from a fixed
set of inputs (the recap, the VTT, the grounding docs).

Two limits of that shape:

1. **The prompt structure is decided up front.** A pipeline pass takes
   a fixed input shape and produces a fixed output shape. The model
   doesn't get to say "actually, given what I just saw, I'd like to
   look up Sildar's loyalty arc before drafting this scene." Each
   pass is a function call, not a conversation.
2. **The whole corpus has to fit in the prompt** *if* you want
   theme-aware output. A 500K-token campaign history doesn't fit in
   any context window. So today you either skip the broad themes or
   you cheap out and ask one pass to do everything from a tiny
   summary.

The pattern this spec proposes: **make Opus the agentic top of the
pipeline**, fetching details on demand instead of receiving them
upfront.

## What's already built that we can use

The good news is the retrieval infrastructure is mostly there. From the
existing CampaignGenerator codebase:

- **`mempalace_client.py`** — JSON-RPC stdio client that lets external
  processes talk to the mempalace MCP server. Black-box integration of
  the mempalace search surface.
- **`rpg_retriever.py`** — three-tiered retrieval orchestrator that
  queries `mem-palace`, `5etools`, and `rpg-library` and returns
  discriminated records. Already supports search/lookup modes; already
  ranks across sources.
- **`fivetools_catalog.py` + `fivetools_ingest.py`** — the canonical
  D&D rules retrieval surface.
- **`scabard_sdk/`** — adventure-module data.
- **mempalace MCP server** — exposes `mempalace_search`,
  `mempalace_search_hierarchical`, drawer-fetch tools.

So we don't need to *build* the retrieval layer. We need to make Opus
*the consumer of it*.

## Architecture

```
                    ┌────────────────────────────────┐
                    │ Opus (Tier 1)                  │
                    │                                │
   thin grounding ►─│  - Decides session theme       │
   (~10K tokens)    │  - Identifies threads to pull  │
                    │  - Drafts structural beats     │
                    │  - Calls retrieval tools as    │
                    │    its thinking evolves        │
                    └────────────┬───────────────────┘
                                 │  tool calls
                                 ▼
                    ┌────────────────────────────────┐
                    │ rpg_retriever (orchestrator)   │
                    │                                │
                    │  ├─► mempalace (Tier 3 / Gemma)│
                    │  │   personal campaign corpus  │
                    │  ├─► 5etools                   │
                    │  │   canonical D&D rules       │
                    │  └─► rpg-library               │
                    │      adventure modules         │
                    └────────────────────────────────┘
```

Opus runs the *thinking*; the retrieval tools handle the *recall*.

## Thin grounding bundle

The starting context for Opus is the smallest amount needed to
identify themes. NOT the corpus. Probably:

| file | what it gives Opus | token budget |
| --- | --- | ---: |
| `campaign_state.md` | current arc, party status, looming threats | ~1K |
| `world_state.md` | active world-level NPCs, geopolitics, factions | ~1.5K |
| `party.md` | PC names, current goals, relationships | ~1K |
| `planning.md` | DM-side notes about what's next | ~1K |
| previous session's enhanced recap | what just happened | ~3K |
| optional: campaign-state diff since N sessions ago | longer-arc trajectory | ~2K |

**Total: ~10K tokens.** That's the prompt's static grounding. Everything
else Opus pulls on demand.

(These four files — `campaign_state`, `world_state`, `party`, `planning`
— are the exact ones the existing `campaign-prep` skill in
`~/.claude/skills/` loads for session prep. The skill's infrastructure
is already the right shape; we're just changing the model behind it
from "whatever is configured" to "Opus + tool-using.")

## Tool surface Opus needs

Opus needs to be able to ask these questions during the session:

| tool | what Opus asks | what it gets back |
| --- | --- | --- |
| `mempalace_search(query, wing=current_campaign)` | "Find passages where Sildar was tested" | top 5–10 drawers, ranked, with source files |
| `mempalace_search_hierarchical(query, wing=...)` | "Tell me about the Helmstone Pact's recent activity" | clustered/summarised drawers |
| `rpg_retriever(query, mode="search")` | "What rules cover stealth in dim light?" | 5etools + rpg-library hits |
| `read_file(path)` | "Show me the canonical text of Cragmaw's stronghold" | file contents (already in MCP filesystem tools) |
| `mempalace_search(query, wing="phandalin", limit=20)` | "All NPC introductions in Phandalin so far" | broad sweep when Opus wants more context |

The point: **Opus decides when each tool is needed.** It might do zero
calls in a simple session prep, or fifty in a complex one. The
token-per-session bill scales with how much detail Opus actually
needed, not with how much detail *exists*.

## Walkthrough: one session-prep run

Step-by-step what a session looks like:

1. **Open Claude Code** in `~/src/campaigns-test/phandalin/`.
2. **Invoke**: `claude --opus` (or `/model opus`) — explicitly call out
   that we want Opus, not Sonnet, for this synthesis-heavy work.
3. **Auto-load grounding** via the existing `campaign-prep` skill (the
   "prep mode" skill in `~/.claude/skills/` that loads the four
   grounding docs).
4. **Opus reads grounding** (~10K tokens) and forms initial hypothesis:
   "Looking at the recap, the party just survived the goblin ambush.
   `world_state.md` says the Cragmaw stronghold is two days away.
   `campaign_state.md` notes Sildar's loyalty has been ambiguous since
   session 7. I think the next session's theme is 'allies tested
   under pressure.'"
5. **Opus pulls details on demand** as it drafts beats:
   - Calls `mempalace_search("Sildar dialogue conflict")` → fetches 5
     past scenes where Sildar's loyalty was relevant.
   - Calls `rpg_retriever("Cragmaw stronghold map and inhabitants")` →
     gets canonical encounter details.
   - Calls `mempalace_search("party reactions to NPC betrayal")` →
     fetches party voice patterns from the corpus.
6. **Opus drafts the session plan** — themes, opening hook, three to
   five beats, a planned moment-of-decision, contingency branches.
   Writes to `~/src/campaigns-test/phandalin/sessions/2026-05-N-prep.md`.
7. **You review / iterate** — push back on specific beats, ask Opus to
   pull more detail on a thread, redraft a particular scene. Each
   round of iteration is another small Opus call + a few more
   retrieval tool calls.

The whole session prep might take Opus 50–80K tokens of total context
(10K grounding + 5–10K outputs + 30–60K of fetched details across
several tool calls). That's **~$1-3 in API cost** for what would
otherwise take a human DM half a day of corpus diving.

## Cost math vs. the alternatives

| approach | Opus tokens per session prep | API cost | quality |
| --- | ---: | ---: | --- |
| Stuff entire corpus into prompt (impossible at >200K) | N/A | N/A | infeasible |
| Pre-summarize corpus into 50K-token "context doc," send every time | ~50K | ~$3-5 | loses detail; static — corpus changes don't propagate |
| **Thin grounding + agentic retrieval (this spec)** | ~60K | ~$1-3 | full corpus visibility, only relevant details retrieved, $0 to refresh corpus |
| Local 14B model with the same agentic pattern | $0 | $0 | possibly too dumb to *decide* good themes, but worth A/B testing |
| Local 14B model with everything in prompt | $0 | $0 | 32K context limit makes this impossible at corpus scale |

The agentic-retrieval pattern wins on cost AND quality because Opus
only fetches what it actually needs to think about, and what it
fetches is always fresh from mempalace.

## Mempalace closet quality as the load-bearing dependency

This whole pattern is gated on **mempalace's closets being good
enough** that Opus's tool calls retrieve the right drawers. If the
search returns garbage, Opus's session plan will be vague and
generic.

The closet quality is exactly what's being tuned right now via:

- [`closet-llm-prefix-cache.md`](./closet-llm-prefix-cache.md) — the
  prompt-prefix optimization to speed up regen.
- [`gemma-vs-qwen-ab.md`](./gemma-vs-qwen-ab.md) — choosing the right
  model to *generate* closets.
- The PR #14 prose-format change (already shipped).
- The pending follow-on: validation queries to confirm the regenerated
  closets actually surface their target files.

**Order of operations:**

1. **First** — confirm mempalace search produces good results on the
   four current validation queries (post the in-flight regen). This
   establishes that the *retrieval layer* is fit for Opus's purposes.
2. **Then** — wire Opus as the agentic orchestrator per this spec.

Don't build the Opus-driven session prep against bad closets.

## Implementation plan

### v1: thin wrapper, manual invocation

The minimum viable version is *don't write much code*. The pieces are:

- The mempalace MCP server is already exposed to Claude Code (the
  hooks fire from your sessions).
- The `campaign-prep` skill in `~/.claude/skills/` already loads the
  four grounding docs.
- Claude Code with `/model opus` selects Opus.
- Claude Code already lets you call MCP tools (`mempalace_search` etc.)
  from within the conversation.

**So the v1 flow is:**

```bash
cd ~/src/campaigns-test/phandalin
claude
# in the session:
/model opus
/campaign-prep
# then start prompting: "Plan next session. Use the grounding I just
# loaded. Pull details from mempalace as needed."
```

That's it. The infrastructure is already there. The new behavior is
just *invoking it deliberately* — pick Opus, lean on tools for detail.

### v2: a `prep` command in spark-cli that wires this up automatically

Once v1 is the comfortable default, the friction is in remembering the
steps. Add a `prep` subcommand to the spark-cli tool described in
[`spark-cli-design.md`](./spark-cli-design.md):

```bash
spark prep ~/src/campaigns-test/phandalin
```

That subcommand would:
1. cd into the campaign dir.
2. Open `claude --model opus`.
3. Auto-invoke `/campaign-prep` to load grounding.
4. Inject a system-prompt-style preamble: "You are the orchestrator
   for D&D session prep. Pull details on demand via mempalace_search /
   rpg_retriever rather than asking the user to recite the corpus."

~30 lines of additional Python. Skip in v1.

### v3: full pipeline replacement

If v1 + v2 prove the pattern, the *next* step is to rewrite
`prep.py`'s pipeline passes themselves to use this same agentic
shape — not just the top-level theme construction. Each pass becomes
"Opus with tool access for its specific subtask" instead of a fixed
prompt template.

This is the largest scope. Defer until v1 has six months of usage data.

## What this unlocks that the current pipeline can't

Things you can do with agentic-retrieval Opus that the fixed-prompt
pipeline can't:

- **"Set up a callback to a moment from session 12 that the players
  might have forgotten."** Opus searches mempalace for session 12,
  finds an emotional NPC moment, and weaves it into the new scene.
  Today, you'd have to hand-feed session 12 into the prompt yourself
  if you remembered it existed.
- **"What unresolved threads should we close this arc?"** Opus reads
  the campaign state, searches for "unresolved," "promised return,"
  "owed favor," and surfaces three threads ranked by urgency. Today
  you'd have to grep the corpus or hold this in your DM brain.
- **"Build me a contingency: what if the party kills Sildar instead?"**
  Opus reads Sildar's history, identifies which factions/NPCs would
  react, drafts the world-state ripple. Today this is the kind of
  thing that costs a session of off-line DM thinking.
- **"Pull every time a player has used 'Detect Magic' creatively."**
  Pure recall. Opus fires off the search, gets the results, weaves
  them into a callback. Today you'd never find this.

The pattern moves the bottleneck from "what can I, the DM, hold in
my head" to "what can I, the DM, ask a smart model to look up."

## Open questions

1. **Does the existing mempalace search surface — even after PR #14's
   prose closets — actually return *good enough* results for Opus's
   queries?** This is the validation gate. If post-regen search
   doesn't surface its LLM-tagged target files in top-5, the agentic
   pattern won't work either: Opus would call the tool and get
   garbage. Resolve via the four validation queries scheduled to run
   after the in-flight regen completes.

2. **Does `rpg_retriever` need a tool-friendly wrapper?** Right now
   it's a Python module used inside CampaignGenerator. To be Opus-
   callable via MCP, it needs to be exposed as a tool. May already be
   via CampaignGenerator's MCP server — check `fivetools_catalog`
   exposure. If not, write a thin MCP wrapper.

3. **What happens if Opus loops on retrieval?** A smart-but-wrong
   Opus might call `mempalace_search("Sildar")` ten times with subtly
   different phrasings. Need to verify that the model is converging
   on a session plan, not wandering. Mitigation: log all tool calls
   and review for first few sessions.

4. **Cost cap.** Per-session ~$1-3 is fine but a runaway agentic
   session could pull dozens of tool calls and rack up $20+. Set a
   token budget on the Claude Code session (`/cost` shows current
   spend) and watch it the first few times.

5. **Should the same pattern apply to *post*-session work
   (consistency check, narration)?** Probably yes — the narration
   pipeline at `session_doc.py` is already pass-based and could each
   pass be an agentic Opus call with retrieval. But that's v3 scope.

## How to pick this up

This spec depends on the closet_llm regen completing successfully and
the four validation queries passing. Once that's confirmed:

> "We have a spec at `~/src/dgx/dnd-session-prep-with-opus.md`. Try
> v1 on next week's Phandalin session prep — open Claude Code with
> `/model opus`, run `/campaign-prep`, ask Opus to plan the session
> using mempalace_search for details. Report whether the resulting
> session plan beats the current prep.py-pipeline output."

That's the validation run. v2 and v3 are deferred until v1 has been
used for a few real sessions.

## Related docs

- [`tiered-llm-workflow.md`](./tiered-llm-workflow.md) — the
  three-tier framework this is a specific instantiation of.
- [`gemma-vs-qwen-ab.md`](./gemma-vs-qwen-ab.md) — which local
  model serves the retrieval layer.
- [`closet-llm-prefix-cache.md`](./closet-llm-prefix-cache.md) —
  speed optimization for the local indexing layer.
- [`finetune-qwen-on-dnd-plan.md`](./finetune-qwen-on-dnd-plan.md) —
  the *other* place a local model wins: stylistic transfer for
  narration. Different from this spec's theme-construction problem.
- [`current-setup.md`](./current-setup.md) — Spark hardware that
  hosts the retrieval layer.
- [`spark-cli-design.md`](./spark-cli-design.md) — workstation tool
  to manage all of the above.
- `~/src/CampaignGenerator/GMASSISTANT_PIPELINE.md` — the current
  fixed-prompt pipeline that this spec proposes augmenting (not
  replacing) at the theme-construction step.
