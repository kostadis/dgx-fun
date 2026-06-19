# DGX Fun

This repository contains experiments, reports, and scripts related to running LLMs and optimizing workflows on local AI hardware (DGX Spark).

## Contents

### LLM Observations & Reports
- **Model Comparisons:** Insights on various models including Gemma 4, Qwen, and Llama.
- **Hardware Calibration:** Reports on DGX Spark calibration and LLM serving learnings.
- **Workflow Design:** Documentation on tiered LLM workflows and D&D session prep.

### Scripts
- **vLLM Spin-up Scripts:** Shell scripts for quickly deploying different models (Gemma 4, Llama 70B, etc.) using vLLM.
- **Testing:** Scripts for testing tool calls and general functionality.

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
