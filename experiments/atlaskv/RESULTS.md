# AtlasKV: standalone Phandelver experiment

Run date: **2026-11-21 PST / 2026-11-22 UTC**. These are local AtlasKV
measurements, kept separate from the paper's published benchmarks and the
earlier whole-system results.

**The released MiniLM AtlasKV bundle failed grounded answer generation in this
trial.** Neither native arm produced a complete grounded factual answer.
Pruning reduced peak CUDA allocation, increased CPU resident memory, and slowed
decoding on this fact table.

The unit under test was the released **Llama 3.1 8B / MiniLM** bundle. All three
arms used the same checkpoint, frozen fact table, questions and greedy decoding.
The text control retrieved 15 facts from that table and supplied them as text.
It is a control inside this AtlasKV experiment, rather than a rerun of the
existing Cognee retrieval workflow.

| Local arm | Complete, grounded answers | Factual latency p50 / p95 | TTFT p50 / p95 | Inter-token latency p50 / p95 |
|---|---:|---:|---:|---:|
| Text-fact control | **3/47** | **3.96 / 9.02 s** | 0.163 / 0.202 s | 83.54 / 84.61 ms |
| AtlasKV, unpruned | **0/47** | **5.70 / 10.07 s** | 0.254 / 0.275 s | 137.75 / 138.49 ms |
| AtlasKV, patched pruning | **0/47** | **8.50 / 14.15 s** | 0.250 / 0.269 s | 174.77 / 185.02 ms |

All 48 factual questions ran in each arm. Quality excludes the predeclared
ambiguous q37; latency includes all 48. The q38 trigger/round rubric is retained.
Codex reviewed every answer against the frozen module evidence, with the text
control's supplied facts also available for support review. This was not a
blinded independent human evaluation. Labels and review notes are retained
privately; the public [aggregate JSON](results-20261121.json) contains metrics
and labels without module passages or answers.

The text control's three complete supported answers were q39, q41 and q44.
It also returned 15 partial answers, 15 incorrect answers and 14 abstentions.
Unpruned AtlasKV returned 46 incorrect answers and one partial answer among
the 47 adjudicable questions; pruning returned 47 incorrect answers. Native
outputs commonly invented names and mechanics, or repeated the question.
All **147 outputs stopped naturally**. None reached its generation limit.

The general question asked how the first part of the adventure, culminating in
Wave Echo Cave, feeds into the second part. Each arm answered once, with no
follow-up agent. The text control took **35.36 s** and gave a generic explanation
with an incorrect chapter handoff. Unpruned AtlasKV took **7.69 s** and supplied
invented locations. Pruned AtlasKV took **5.93 s** and echoed the question.
None recovered the actual transition or a grounded distinction between GM and
character knowledge. The shorter native responses do not represent useful
answering speed.

| Local arm | Worker peak CUDA allocation | Worker peak process RSS | Minimum host available memory |
|---|---:|---:|---:|
| Text-fact control | 16.67 GiB | 2.30 GiB | 36.42 GiB |
| AtlasKV, unpruned | 20.24 GiB | 2.18 GiB | 31.51 GiB |
| AtlasKV, patched pruning | 16.22 GiB | 7.65 GiB | 32.58 GiB |

These are query-phase measurements across all 49 questions per arm on a
121.69 GiB unified-memory Spark, with Clef resident. CUDA allocation and RSS
are separate views and should not be added as a total. Pruning lowered observed
peak CUDA allocation by **4.02 GiB**, while minimum whole-host headroom improved
by about **1.06 GiB** and process RSS increased. CPU offload therefore provides
a smaller system-memory benefit here than the CUDA figure alone suggests.
The fixed arm order, shared device and single run limit precision.

The fact table contains **21,091 records** exported from per-chunk Cognee
extraction responses spanning all **759 chunks** of the module. Every record
retains source lineage. These are extracted assertions rather than verified
gold facts. There are 2,729 keys with multiple values, and 13,501 values contain
at most five words. Such a representation can lose conditions, scope and
context. GM/player visibility remains metadata rather than an enforced
attention policy. The weak text control is relevant to interpreting this
particular export and reader together; it does not isolate an AtlasKV-only
effect or score the larger system.

Offline sentence encoding and hierarchy construction took **200.93 s**. The
hierarchy had 764 intermediate and 28 root clusters, without encoder truncation.
Top-k 128/64/16 consequently keeps all roots at this scale, then selects
intermediate and leaf candidates. Preparing the projected knowledge took
**0.61 s unpruned / 44.06 s pruned**. Factual query totals were
**225.28 / 289.02 / 428.25 s** for text, unpruned and pruned respectively.
The completed GPU worker took **19 min 12 s**, including loading, preflights,
knowledge preparation and all three arms. Downloads, relation rewriting and
offline encoding are separate preparation costs.

Two corrections were required before the measured run:

1. The native constructor loaded the inner Llama with `sdpa` despite the outer
   `eager` setting. With attention output disabled, it omitted the causal mask
   while computing attention manually. The runner now enforces eager masking
   in both configurations. Synthetic text-evidence use failed before this
   correction and passed afterward with the same weights. The stopped pilot
   is excluded from every table above.
2. The upstream pruning code could pair selected keys with values indexed from
   the wrong candidate positions. The [index patch](pruning-index.patch) and
   [regression check](check_pruning.py) preserve intermediate and leaf mappings
   and singleton index shape. This arm is explicitly **patched**.

Checkpoint loading had no missing, unexpected or mismatched weights. The trained
encoder's frozen MiniLM weights exactly matched the encoder used for offline
embeddings. Even after the mask correction, a three-fact synthetic memory probe
failed: the stored rule required an ochre permit, while the generated answer
described a fishing permit. This establishes a generation failure in the tested
configuration; it does not separate attention ranking from value reconstruction.

The [AtlasKV paper](https://arxiv.org/abs/2510.17934v2) reports different evaluation
settings: [Appendix A.2](https://arxiv.org/pdf/2510.17934v2#page=16) uses MiniLM for
knowledge-grounding attention ranking and `text-embedding-3-large` for generation
relevance. This local trial evaluates the released MiniLM bundle on adventure
answers. Its outcome does not reproduce or replace those published results.

Pins, protocol, isolation and scripts are in the [experiment README](README.md).
The [operations receipt](operations-20261121.json) records worker exit and service
recovery. Full prompts, answers, embeddings and review notes remain under
ignored `raw/phandelver-20261121/` and the private Spark trial directory.
Original extraction captures remain in the existing private Cognee data,
referenced by the source manifest. Earlier result documents were left
unchanged by this experiment.
