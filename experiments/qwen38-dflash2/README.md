# Qwen3.8-Flash-Next DFlash 2 on DGX Spark

**Historical trial state, 2026-11-07 08:19 PST:** DFlash 2 was validated at
256K and left running on spark1 :8002, **262144 total context / 32000 default
output allowance / one sequence**. Original vLLM Qwen is stopped and
retained; :8001 is down. spark2 is untouched. No clients were repointed
and no boot startup was installed.

That deployment was superseded on 2026-11-11: DFlash was stopped and the retained
vLLM Qwen container restored. The scripts and evidence below describe the trial;
see [current setup](../../current-setup.md) for later deployment snapshots.

[Results and limitations](../../qwen38-dflash2-observations.md): all short
checks and five-fact retrieval from **224994 input tokens** passed with
`max_tokens: 32000`. The answer ended at 124 tokens, so full 32K generation
was not tested. Input and output share the context: a 32000-token response
allows at most 230144 prompt tokens including template overhead.
Warm 8K DFlash was 22.3% faster than MTP over three prompts (geometric mean).

## Pinned configuration

- Host: spark / gx10-46ea, GB10, ARM64, CUDA 13.0, sm_121.
- Separate remote directory: `~/qwen38-dflash-trial/`.
- Runtime: ggml-org/llama.cpp commit `ca89968a23a13e571a6b802a9891d22763153b61`.
- Build: native Release / CUDA, architecture 121, two parallel jobs.
- Target: ggml-org/Qwen3.8-Flash-Next-GGUF, IQ4_NL, two shards.
- Model repository revision: `91fb00353e766eb43944ed223665ea4fb87dad47`.
- MTP sidecar: `mtp-Qwen3.8-Flash-Next-Q4_0.gguf`.
- External drafter: `dflash-Qwen3.8-Flash-Next-Q4_0.gguf`.
- All files downloaded with size and SHA256 checks before launch.
- The large PLE embedding table is marked lazy-read in this runtime; it is not intended to be pinned as a complete GPU allocation.

## Comparison method

Captured the existing vLLM MTP-2 endpoint as an operational baseline. Used the same IQ4_NL target and llama.cpp build to compare ordinary generation, native MTP, and external DFlash 2. The vLLM-vs-llama.cpp comparison changes quantization and runtime, so it cannot isolate DFlash's benefit. The MTP-vs-DFlash comparison within llama.cpp can isolate the drafting choice more closely.

Start at 8K context, one sequence, bounded prefill batch, with a global available-memory guard. Preserve the current `qwen38-flash` container and source builds. The intended successful end state is a running validated DFlash server; failure or memory pressure will be handled with the retained Qwen deployment available for recovery.

Sources: [target and sidecars](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF), [drafter usage and reported measurements](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF).

## Running and recovering

On spark, after ensuring no other native trial is running:

```bash
docker stop qwen38-flash
nohup python3 ~/qwen38-dflash-trial/run-server.py --mode dflash --context 262144 --output-tokens 32000 \
  > ~/qwen38-dflash-trial/dflash-launch.log 2>&1 </dev/null &
python3 ~/qwen38-dflash-trial/wait-ready.py
```

The API is `http://192.168.1.147:8002/v1`, model
`qwen3.8-flash-next-gguf`. The guard writes `active.json`, per-run
`launch.json`, `memory.jsonl`, `ready.json`, `server.log`, and eventually
`exit.json`. It enforces 100 GiB CPU cgroup memory/no cgroup swap and stops
the child process group if global MemAvailable drops below 8 GiB.
No reboot startup is installed. Only one native trial should run at once.

To recover the original vLLM service, terminate the `guard_pid` from
`active.json`, wait for that run's `exit.json`, then start
`docker start qwen38-flash`. Verify :8001 `/health` and `/v1/models`.
Do not start both runtimes concurrently.

`benchmark.py` records exact prompts, responses, SSE events, usage, TTFT,
wall time, and native timings. Five arithmetic/copy checks gate the three
384-token performance prompts. `compare.py` runs plain-repeat, MTP plus
repeat, then DFlash plus repeat, leaving successful DFlash running.
Caching is disabled and native timings must show `cache_n: 0`.
The first and repeated passes are retained because the initial plain
pass showed a clear warm-up effect. These are small trials, not a broad
quality or throughput evaluation. No clients have been repointed.


The controlled comparison used 8192 context in all modes. `finalize-context.py`
then raised DFlash to 32768, reran the short suite and a 29,987-token
retrieval test. `test-context.py` writes exact prompts/results under the
remote `context-tests/` directory. That 32K configuration was subsequently superseded by the 256K deployment
described below. Stopping a benchmark does not stop its guarded server.
Those 32K snapshots are historical. Current 256K launch/process evidence is
in `evidence/final-state-256k.json`; LAN verification is in
`evidence/final-lan-verification-256k.json`.

Controller scripts are the one-off orchestration used for this trial.
Their completion markers/results belong to this run; use a fresh result
namespace before rerunning a full comparison, and stop the live native
server deliberately before launching another.

## 256K validation

The launcher now accepts `--output-tokens` (default 32000), passed as
`-n` to llama.cpp. Explicit request `max_tokens` overrides this server
fallback. `/props` lists the per-request sentinel `-1`; the server source
selects the CLI fallback when that sentinel is unchanged.

Run the retained large-context retrieval method on an idle server:

```bash
python3 ~/qwen38-dflash-trial/test-context-large.py \
  --context 262144 --target 225000 --output-tokens 32000
```

Results: `evidence/262144-result.json`, exact input in
`evidence/262144-prompt.txt`, and `evidence/256k-validation.log`.
The frozen 8K and 32K results are retained independently. Current dgxlib
behavior uses max_tokens 32000, read timeout 3600 s, idle timeout 1800 s,
and max concurrency 1 for the same served model ID.
