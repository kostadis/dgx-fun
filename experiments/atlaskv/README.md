# AtlasKV on the Phandelver Cognee corpus

This is an isolated experiment based on
[AtlasKV, ICLR 2026](https://arxiv.org/abs/2510.17934), comparing three knowledge
access methods with the same released Llama 3.1 8B checkpoint.

The completed [standalone results](RESULTS.md) and
[aggregate metrics](results-20261121.json) cover this AtlasKV experiment only.
The released MiniLM native arms produced 0/47 complete grounded answers;
the same-backbone text-fact control produced 3/47. Paper-reported benchmarks
and earlier whole-system results remain separate.

| Arm | Knowledge access |
|---|---|
| `rag` | Cosine retrieval of 15 facts using the matching MiniLM encoder, supplied as text |
| `atlas` | All fact keys/values supplied through native AtlasKV attention, without pruning |
| `pruned` | Native three-level knowledge attention with top-k 128/64/16 and the indexing correction below |

The RAG arm uses the same native model with knowledge injection disabled. It is
a control for knowledge access, not an optimized vLLM speed baseline. None of
the arms trains on Phandelver or on the held-out questions. The general module
question is a single answer in each arm, without an additional research agent.

## Frozen inputs and checkpoints

- Source: `HKUST-KnowComp/AtlasKV`, commit
  `61e3abc1fb09fbfb426cfc847c4c26c5e94bcaeb`.
- Model: `HaoyuHuang2/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000`,
  revision `0a4572b85ffd818f5c3b22a407e428bb2d542c27`.
- Matching learned encoder: the corresponding `_encoder` repository, revision
  `30ef42250325b656ec977792cd6bf87ad19375e7`.
- MiniLM: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct`, revision
  `a2856192dd7c25b842431f39c179a6c2c2f627d1`.
- Runtime: private Python 3.12 environment over the cached ARM64 CUDA 13 image
  `sha256:f4612b72c529a2e899c008b4ecc970445352cc004f38bbe2afe92507ca43167d`,
  with Torch 2.13, Transformers 4.46 and Sentence Transformers 3.3.1.

`prepare.py` exports per-chunk extraction responses from the existing corrected
adaptive index, before global entity descriptions were merged. All 759 chunks
retain exact request/capture lineage. The export contains 16,291 assertions;
forward/reverse relation keys and descriptions deduplicate to 21,091 records.
Relation nouns are prepared by the local Qwen service using only relation names.
The benchmark questions and answer keys are never inputs to this preparation.

These are model-extracted assertions, not verified gold facts. Unresolved
textual endpoint labels are retained and flagged; no descriptions are invented
for them. Conflicting values remain separate records. Every record retains its
source references and is marked as GM module reference material, with character
knowledge unassessed. Those visibility labels are metadata, not an enforcement
mechanism inside the learned attention.

`encode.py` uses the matching released MiniLM representation and the upstream
UMAP/GMM parameters: 64 dimensions, 30 neighbors, two GMM levels, two initializations,
50 iterations and seed 42. Empty hard-assignment components are removed before
pooled centroids are published, with membership maps preserved. Encoder length
limits and any truncation are recorded.

This table produces 764 intermediate clusters and 28 root clusters. Top-k
128 therefore retains all root clusters at this scale; intermediate and leaf
selection still reduce the candidates to 64 and 16 respectively.

## Pruning correction

The pinned upstream code selects intermediate keys from a candidate subset but
maps their positions through the full index list. It also indexes the full value
array using positions within the pruned leaf subset. A selected key can thus be
paired with another fact's value. Singleton selection also collapses index rank.

[pruning-index.patch](pruning-index.patch) preserves candidate-to-original index
mapping and keeps singleton indices as vectors. [check_pruning.py](check_pruning.py)
executes the actual upstream expressions on non-contiguous candidates: upstream
selects leaf value/intermediate ID 2/2 instead of the intended 11/18; patched
expressions select 11/18. The corrected arm must be labeled as patched, rather
than presented as an unchanged reproduction of published code.

## Causal-mask configuration

The native causal-LM constructor loads its inner Llama model without forwarding
the outer attention backend. In this runtime, requesting outer `eager` still
loads the inner model with `sdpa`. With `output_attentions=False`, that policy
can return no causal mask, although AtlasKV computes attention with manual eager
operations. The first stopped pilot used this inconsistent configuration and
is excluded from benchmark results.

The runner explicitly sets both configurations to `eager` and records their
loaded/effective values. [check_attention.py](check_attention.py) exercises the
actual upstream mask method with tiny tensors and verifies that future tokens
are blocked. Synthetic generation failed with the missing mask and answered
the requested facts after the setting was corrected, using the same weights.
The default preflight checks both arithmetic and text-evidence use. It also
checks that the trained encoder's frozen sentence-model weights exactly match
the sentence model used to precompute the fact embeddings.

## Measurement protocol

All arms use greedy decoding, seed 604, batch one and fresh sequence KV state
per question. The original 48 factual questions retain their 120-word instruction;
the module-transition question has a 600-word instruction. Generation allowances
are 512 and 1,200 tokens respectively, with an 8,192-token prompt limit. Output
length termination is recorded as such. The publication's `kb_scale_factor=100`
evaluation setting and injection every three layers are retained.

Question latency includes retrieval/query embedding where applicable and full
generation. Loading, offline encoding, clustering and memory preparation are
recorded separately. Token streaming timestamps include the first generated
token, including special tokens, rather than only visible text.

Memory records host available memory and swap, process RSS, container memory
accounting and this worker's CUDA allocator. These must be interpreted together
on unified-memory GB10; CUDA allocation alone is not total system memory.
Runs occur in fixed arm order on a shared device, without repeated-run statistics.

Source review retains the original q37 ambiguity exclusion and q38 trigger/round
rubric. Fact-selection provenance and attention scores do not establish answer
correctness or causality. Full inputs, answers, captures and source review stay
under ignored `raw/phandelver-20261121/` and the private Spark trial directory.

## Isolation

CPU preparation uses a container without GPU access. Only after preparation is
complete does the Spark-side controller temporarily stop Inkling, run the native
worker with a 40 GiB memory bound, and restore the retained Inkling container.
Clef stays running, and Spark 1's primary Qwen is unchanged. The worker has no
network access and exposes no serving endpoint. Source, cache and Python packages
are private to `spark2:~/atlaskv-trial/`; Cognee's installation and saved indexes
are read-only inputs.

The paper uses MiniLM for its principal attention-ranking experiments and the
larger OpenAI encoder for its answer-relevance experiments (Appendix A.2).
This local-only trial evaluates the available MiniLM checkpoint, including its
ability to reconstruct factual answers; it does not reproduce the larger-encoder
generation setting. A failed synthetic answer is retained as a result rather
than silently treated as evidence of correct knowledge access.
