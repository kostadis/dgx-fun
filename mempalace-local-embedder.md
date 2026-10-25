# Mempalace on a local embedder

Oct 25, 2026 · @kostadis

Mempalace runs here on a fork (`kostadis-dev`) because upstream assumed a single machine that embeds its own text and stores it in ChromaDB. Ours embeds on a separate GPU box and stores vectors in turbovecdb.

## What upstream assumed

Upstream mempalace embeds on the machine that runs it, with a bundled ONNX model, and keeps drawers in ChromaDB. That works on a laptop. It does not fit a setup where the GPUs sit on a DGX Spark, the campaign checkouts sit on a Linux VM, and several campaign palaces share one install.

Three assumptions had to change:

- **Where embedding runs.** The ONNX default runs on the CPU of whatever machine mines. We wanted it on the Spark's GPU, without installing CUDA wheels on the mining host.
- **What stores the vectors.** ChromaDB was a hard import. A turbovec-only install still pulled in chromadb and broke when it was absent.
- **How many palaces one process serves.** Several code paths quietly fell back to a single default palace. Campaign palaces leaked into each other and into the chat palace.

## How embedding works on this host

Every vector in every campaign palace comes from one local model: Ollama serving `qwen3-embedding:0.6b` (1024 dimensions). Nothing leaves the local network.

- **Declared once.** `~/.config/hypostasis/hypostasis.yaml` sets `MEMPALACE_BACKEND: turbovec`, `MEMPALACE_EMBEDDING_PROVIDER: ollama`, the model, and the endpoint `http://192.168.56.1:11434`. That address is the VM's host-only network, so Ollama runs on the machine that hosts this VM. hypostasis renders these values into every component; no tool picks its own embedder.
- **Mining.** The miner calls `get_embedding_function()`, which the provider router resolves to `OllamaEmbeddingFunction`. It posts chunks to Ollama's `/api/embed` in sub-batches of 64, then writes the vectors to turbovecdb.
- **Searching.** Each turbovec collection is opened with a lazy embedder that resolves the same `get_embedding_function()`. Queries and stored drawers are always embedded by the same model.
- **Identity.** Each provider reports its own name. Switching the model or provider in config therefore raises an embedder-identity error, instead of silently mixing vectors from two models in one palace.

## What we changed in mempalace

The fork carries about a dozen changes on `kostadis-dev`. Each one removes a single-machine or single-palace assumption. Upstream releases v3.3.6 through v3.10.0 were merged in on top of them.

| Date | Change | Ref | Why |
| --- | --- | --- | --- |
| 2026-10-25 | Sub-batch Ollama `/api/embed` requests (default 64) | #57 / PR #58 | One 1000-chunk request crashed Ollama's model runner on large files |
| 2026-10-24 | `mine --skips-json` lists every file that was not mined | PR #56 | mneme's corpus check needs skips as data, not parsed stdout |
| 2026-10-23 | Corpus verbs: `mine --files-from`, `sources --json`, `forget` | PR #55 | mneme mines an explicit file list instead of sweeping folders |
| 2026-10-04 | Per-palace isolation: cross-palace reads use the target's backend, `default_palace` honoured, `palace=` on every read tool, per-palace tunnels and entity registry, hook writes routed to the chat palace | #41, #43, #44, #45, #48, #51 | Campaign palaces leaked into each other and into the chat palace |
| 2026-07-01 | Chroma backend made lazy and optional | #20 | A turbovec-only install still imported chromadb |
| 2026-05-30 | turbovec storage backend, later a thin adapter over turbovecdb | — | Compact vector storage without ChromaDB's HNSW and SQLite stack |
| 2026-05-10 | Remote embedding providers: Ollama and OpenAI-compatible, with keep-alive connections | #6 | Embed on a GPU box without installing CUDA wheels on the mining host |

## Still rough

- **Log noise.** Every mined file logs "vectors= bypasses the configured embedder". turbovecdb warns because the miner passes precomputed vectors, but those vectors come from the same model the collection queries with. Filed as [mempalace#59](https://github.com/kostadis/mempalace/issues/59).
- **Embedder identity is stored but never read.** turbovecdb records the embedder function's name (embed, for every palace here), not the model. mempalace's turbovec adapter never reads or writes it. So every open of a populated palace warns, and set-embedder cannot clear the warning ([mempalace#33](https://github.com/kostadis/mempalace/issues/33)).
- **Stale remote.** GitHub reports the repo moved to `kostadis/mempalace`, but the local remote URL still uses the old name.

The lesson: switching to a local model was one config line. The work went into the assumptions around it: where embedding runs, how large a request the server tolerates, and which palace a call reaches.
