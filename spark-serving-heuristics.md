# Spark serving heuristics — distilled from session memory

Hard-won, **transferable** rules for serving LLMs on the GB10 Sparks — extracted
from Claude's private project memory into the repo so they persist and are
shareable (2026-06-20). These are the *lessons*, not the live config — for
what's running right now see `current-setup.md`; for the reservation math see
`gpu-reservation-and-kv-tradeoffs.md`.

Each entry: the rule, the why, and where the detail lives (repo doc and/or the
memory slug it came from). Skim this before any new serving experiment.

---

## Memory & reservation

- **Reserve `--gpu-memory-utilization` at 0.80, not 0.88.** Unified memory: the
  reservation steals host RAM; 0.88 left ~15 GB host, starved sshd's fork, and
  wedged the box (reboot, 2026-06-20). Full reasoning in
  `gpu-reservation-and-kv-tradeoffs.md`; rule in `feedback_gpu_util_080_default`.

- **Size context from the *measured* KV pool, not slot guesses.** Read
  `GPU KV cache size` from the vLLM startup log before bumping `--max-model-len`.
  On hybrid GDN/Mamba+full-attn models only the few full-attention layers carry
  KV (Qwen3-Next pool ≈ 2.53M tokens at 0.88), so big context is cheap — a dense
  model would be far costlier per token. (`feedback_size_context_by_kv_pool`)

- **Decode is bandwidth-bound; KV above your real concurrency is wasted.** At
  273 GB/s, reserving more KV than the bus can feed at your latency floor just
  steals host RAM. Size util to hold `weights + overhead + seqs×ctx×kv/tok +
  ~25GB host`, then stop. (`gpu-reservation-and-kv-tradeoffs.md`)

- **Diagnose host starvation by signature:** ping + an already-running container
  stay up, while the new port refuses connection and sshd **times out during
  banner exchange**. That's host-RAM starvation → lower util (or reboot). A GPU
  OOM is different — container exits with a CUDA error, box stays responsive.

- **One vLLM chat slot saturates spark1; a second container OOMs.** At chat util
  0.88 only ~2.4 GB free. Ollama can't co-host either — it falls back to CPU at
  ~3 tok/s. Spec the small/embed container *first* so chat sees free GPU at boot.

## Benchmarking & workload shape

- **Always measure prefill AND decode separately.** MoE wins prefill
  (compute-bound, few active params); AWQ / spec-decode win decode
  (bandwidth-bound). Decode-only benchmarks hide the production gap.
  (`feedback_llm_bench_prefill_vs_decode`)

- **This user's workloads are read-heavy** — lots of context in, less out.
  Prefer MoE / smaller-dense for prefill speed; lead recommendations with prefill
  characteristics, not decode tok/s. (`user_workflow_read_heavy`)

- **…but match the accelerator to the bound.** Spec decode / MTP helps *decode*
  (long structured-output render — e.g. pdf-translators), NOT prefill-bound read
  jobs. Pair `(model capability) × (call intent)`: don't add MTP to speed up a
  prefill-dominated task. (Session note, 2026-06-20; `todo_speculative_decoding`)

## Model & parser selection

- **Avoid raw `<think>` leaks into clients that don't strip them.** Prefer
  Instruct over Thinking variants for llm_wiki/CampaignGenerator; if you must run
  a reasoning model, wire a reasoning parser that routes traces to
  `reasoning_content` (not the non-standard `reasoning` field — that leak got
  Nemotron rejected and is silently dropped by opencode).
  (`project_nemotron_rejected`, `todo_nano_v3_reasoning_leak`)

- **Qwen3-Coder wraps its *entire* answer in `<think>`** — with the `qwen3`
  reasoning parser active, `content` comes back null. Serve it WITHOUT a
  reasoning parser so raw output stays in `content`. (`current-setup.md` §3)

## Quantization & kernels (GB10 / sm_121)

- **NVFP4 runs on real CUTLASS FP4 kernels on GB10** — validated via the
  Nemotron-3-Super NVFP4 experiment (Nemotron-H loads at 120B). It's a real
  decode-bandwidth lever, but on a *dense* model it reads all weights/token, so a
  3B-active MoE still decodes faster. (`project_nemotron3_super_nvfp4`)

- **Prefer pre-quant vLLM paths over TensorRT-LLM engine builds for novel
  hybrids.** TRT-LLM likely lacks Gated DeltaNet (Qwen3-Next) support and a
  stack switch loses the `gemma4`/`hermes` vLLM tool parsers the clients depend
  on. MTP is available *in vLLM* (`qwen3_next_mtp`) — no TRT-LLM needed.
  (Session NVFP4/MTP analysis, 2026-06-20)

- **TurboQuant KV is a calibration choice, not a win here.** On a hybrid only the
  full-attn layers carry KV, so the memory saved is small while the Hadamard/FA2
  overhead is paid in full on the prefill path. Plain fp8 KV is the better
  default. (`turboquant-observations.md`)

## Embeddings & retrieval

- **qwen3-embedding:0.6b beats nomic decisively** (hit@5 57.5% vs 44.5%, ~+27%);
  adopted. 1024-dim, instruction-aware. (`project_embedding_qwen3_upgrade`)

- **Evaluate quantized vector indexes end-to-end (hit@k on NL queries), not on
  index recall@k vs exact.** Index recall overstates quantization cost — turbovec
  4-bit tied ChromaDB end-to-end despite a 3–4 pt pure-vector gap.
  (`feedback_index_recall_vs_endtoend`)

## Networking & environment (bit-you-once gotchas)

- **After any network change, test the real TCP path — not just ping/RDMA.**
  ICMP ping and `ib_write_bw` (RDMA) both pass on a broken IP config; verify the
  actual TCP path and re-read live `ip addr`. (`feedback_test_tcp_not_just_ping_rdma`)

- **`.profile` env vars need `export` to reach ssh-launched scripts.** A
  single-shell test lies about child-process reachability — verify with a
  grandchild process when an env var must reach an ssh-launched script (e.g.
  `HF_TOKEN`). (`feedback_profile_export_keyword`)

- **Use `ssh spark` / `ssh spark2` (config aliases, key auth).** `gx10-46ea`
  doesn't resolve; the raw IP falls back to password. But for *endpoints* in
  commands/configs, always use the IPs (`192.168.1.147` / `.121`) — the
  hostnames don't resolve in WSL2. (`reference_spark_ssh`, `reference_spark2`)

- **Cross-box TP=2 works over the direct RoCE cable** despite modest bandwidth —
  the win is low all-reduce *latency* (point-to-point DAC), not plentiful BW.
  PP=2 is the next decode experiment. (`reference_spark_2box_vllm`,
  `reference_deepseek_v4_flash_2box`)
