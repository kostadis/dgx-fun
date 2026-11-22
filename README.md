# DGX Fun

This repository contains experiments, reports, and scripts related to running LLMs and optimizing workflows on local AI hardware (DGX Spark).

## Contents

### LLM Observations & Reports
- **[Hibana on DGX Spark Results](hibana-spark-results.md):** Shadow comparison against pinned vLLM, including real Cognee ingestion, long-context replays, quality findings and crash recovery.
- **[Phandelver and Below Chunking Results](phandelver-and-below-chunking-results.md):** A 48-question comparison of fixed, recursive and adaptive chunking, including context-budget sensitivity and answer-grounding checks.
- **[Phandelver in Cognee: Chunking and Sequential Retrieval](phandelver-cognee-agent-results.md):** Full Cognee trial, Qwen/Haiku/Sonnet/Inkling agent results, and the Inkling and Hibana fast factual and narrative comparisons, including quality, timing and costs.
- **[Temple of Elemental Evil Chunking Pilot](adaptive-chunking-toee-results.md):** The initial 40-question trial, with retrieval measurements, answer checks and limitations.
- **Model Comparisons:** Insights on various models including Gemma 4, Qwen, and Llama.
- **Hardware Calibration:** Reports on DGX Spark calibration and LLM serving learnings.
- **Workflow Design:** Documentation on tiered LLM workflows and D&D session prep.

### Scripts
- **vLLM Spin-up Scripts:** Shell scripts for quickly deploying different models (Gemma 4, Llama 70B, etc.) using vLLM.
- **Testing:** Scripts for testing tool calls and general functionality.
- **[Inkling 2 Mini on Spark 2](experiments/inkling/README.md):** The selected [native vLLM service](experiments/inkling/NATIVE.md), no-thinking client controls, and separate native/Unsloth DSpark measurements. The [paired native test](experiments/inkling/native-thinking/RESULTS.md) cut factual median latency from 10.38 to 4.51 seconds with 44/47 complete answers in both arms and one additional unsupported claim without thinking.
- **[D&D Chunking Experiments](experiments/adaptive-chunking/README.md):** Frozen-source preparation, local reference expansion, retrieval comparisons, answer checks and aggregate result exports.

### Library
- **[`dgxlib/`](dgxlib/README.md):** Installable Python package owning per-model
  request behavior (thinking/timeout/max_tokens registry + model discovery), so a
  model swap is a one-line edit to `dgxlib/models.yaml` rather than code surgery in
  callers. Consumed by CampaignGenerator and mytools. See
  [`dgxlib/ARCHITECTURE.md`](dgxlib/ARCHITECTURE.md).

## Usage

Most deployment scripts can be run directly:

```bash
./spin-up-vllm-gemma4-26b-moe.sh
```
