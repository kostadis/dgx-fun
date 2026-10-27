# Flash-1017 near-600K context test

Completed 2026-10-20 23:36:49 UTC on the existing two-Spark TP=2 deployment.
No serving settings changed. Model: Flash-1017 EAQ EXL3 2.9 bpw, served as
`DeepSeek-v4.1-Flash-EXL3`, vLLM `0.1.dev20904+g179dd0fa9`.

| Measurement | Result |
|---|---:|
| Exact prompt tokens | 599,488 |
| Output budget | 512 |
| Actual output / total tokens | 82 / 599,570 |
| Cached input tokens | 0 |
| Exact retrieval | 5 / 5 |
| Time to first content token | 663.15 s |
| End-to-end time | 664.53 s |
| Effective input-to-first-token rate | 904 tokens/s |
| Minimum available memory, head / worker | 4.84 / 6.80 GiB |
| Preemptions / restarts / OOM flags | 0 / 0 / 0 |

`test.py` builds randomized word records and inserts random exact-match values
at approximately 5%, 25%, 50%, 75%, and 95% of the records. It calibrates the
chat prompt using the deployed `/tokenize` endpoint, then streams one request
with temperature zero and thinking disabled. Response usage independently
confirmed the token count. All five values matched and finish reason was `stop`.

Ollama embeddings were already unloaded and remained unloaded. Memory was
sampled every three seconds through both SSH aliases; swap usage did not grow.
Both containers remained running with zero restarts and no OOM flag. Neither
rank logged ERROR, Traceback, or out-of-memory matches during the test.
The post-test arithmetic request returned 323 in 0.30 seconds; health was 200.

This is a single-request capacity and synthetic retrieval test. It does not
establish complex long-document reasoning quality, concurrent-request capacity,
or memory safety with embeddings resident. The budget reached exactly 600,000;
the model stopped before consuming its entire output allowance.

Evidence: `metadata.json`, `result.json`, `events.jsonl`, `post-health.json`,
`run.log`, and both `*-memory.tsv` files. The larger exact `request.json` and
metrics snapshots are retained locally and ignored by Git. Request SHA256 is
in metadata. Rerunning `python3 test.py` overwrites evidence in this directory;
copy the directory first if preserving a run.
