# Clef / Clef-flash decision-model server for the DGX Spark (GB10, sm_121, aarch64).
#
# Base is the vLLM image ONLY for its torch build: 2.11.0+cu130 is the exact torch
# the Clef release was tested with, and it is already proven on GB10. vLLM itself
# is not used — the joint schema head is custom code that vLLM cannot load.
FROM vllm/vllm-openai:latest

# transformers 5.10.2: the release's tested version (the base ships 5.8.1).
# flash-linear-attention: the Triton gated-delta-rule kernels for Qwen3.5's linear
# attention layers; without it transformers falls back to a slow torch loop.
# fla is split into flash-linear-attention + fla-core (kernels); --no-deps on both so pip cannot replace the base image's torch.
RUN pip install --no-cache-dir "transformers==5.10.2" pillow \
 && pip install --no-cache-dir --no-deps flash-linear-attention fla-core \
 && python3 -c "import torch, transformers; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print(torch.__version__, transformers.__version__, 'fla ok')"

COPY server.py /app/server.py
ENTRYPOINT ["python3", "/app/server.py"]
