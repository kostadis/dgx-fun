# ~/src/dgx — Claude instructions

This repo is the working tree for DGX Spark experiments: spin-up
scripts, observation reports, and design docs for running LLMs on the
Spark (`192.168.1.147`, hostname `gx10-46ea`).

## The hard rule: keep `current-setup.md` honest

`current-setup.md` is the inventory of **what is actually running on
the Spark right now** — ports, containers, models, flags, VRAM budget,
client-side config. It is the file the user reaches for when something
breaks at 11pm or when the box gets wiped.

A stale `current-setup.md` is worse than no `current-setup.md`.

### When you must update `current-setup.md`

Update it in the **same change** that does any of the following:

- Swaps the model served by `vllm-chat` (e.g. Gemma 4 → Llama 70B,
  Qwen 14B → Gemma 4, etc.) — this includes running any of the
  `spin-up-vllm-*.sh` scripts that replace the container.
- Adds, removes, or modifies vLLM flags on a running container
  (`--max-model-len`, `--gpu-memory-utilization`, `--dtype`,
  `--enable-auto-tool-choice`, `--tool-call-parser`,
  `--max-num-batched-tokens`, etc.).
- Adds a new long-running container or systemd service on the Spark
  (e.g. a third vLLM sidecar, a new embeddings model, a finetune
  server).
- Changes a port assignment.
- Changes the Ollama service config or `override.conf` env vars.
- Changes any client config that points at the Spark
  (MemPalace, llm_wiki, CampaignGenerator, opencode) in a way that
  reflects a server-side change.
- Changes the LAN IP, hostname, or anything in the Hardware section.

### What to update

At minimum, sync these sections:

- §"Ports in use" table
- §"VRAM budget (steady state)" table
- The relevant numbered service section (§2 vllm-embed, §3 vllm-chat,
  etc.) — model id, run command, "Why these flags", measured behaviour
- §6 "Client-side configuration" if the model id consumed by clients
  changed
- §7 "Rebuild-from-scratch order" if the spin-up sequence changed
- The "Snapshot ... as of YYYY-MM-DD" line at the top — bump it to
  today

### What does NOT need an update

- Pure observation/report doc edits (`gemma4-26b-moe-observations.md`,
  `model-comparisons.md`, `spark-llm-serving-learnings.md`, etc.)
- New experiment plans / runbooks that haven't been deployed yet
- Edits to spin-up scripts that aren't currently the active script for
  `vllm-chat`

If you're unsure whether a change is "live" or just a plan, check
`docker ps` against the Spark or `curl -sS http://192.168.1.147:8001/v1/models`
before editing. The deployed state, not the script content, is what
`current-setup.md` documents.

### Verifying before you edit

Before changing `current-setup.md`, confirm reality on the box:

```bash
# What models are actually being served?
curl -sS http://192.168.1.147:8000/v1/models   # vllm-embed
curl -sS http://192.168.1.147:8001/v1/models   # vllm-chat
curl -sS http://192.168.1.147:11434/api/tags   # ollama pulled
curl -sS http://192.168.1.147:11434/api/ps     # ollama loaded
```

If a section of `current-setup.md` disagrees with the live response,
the live response wins — update the doc.

## Other notes for this repo

- The `spin-up-vllm-*.sh` scripts in this directory all target the
  `vllm-chat` container slot on port 8001. Only one can be active at
  a time. The active one's flags must match §3 of `current-setup.md`.
- Observation reports (`*-observations.md`, `*-learnings.md`,
  `*-comparisons.md`) are append-only experiment logs, not specs.
  Don't rewrite history in them; add new sections instead.
- The user is calibrating local AI hardware on purpose. Suboptimal
  configurations are expected — see the "Local AI Hardware Exploration"
  rule in the global CLAUDE.md.
