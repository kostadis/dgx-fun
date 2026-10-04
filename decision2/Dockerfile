# vLLM Semantic Router model runtime (Decision 2.0) on the GB10.
# Base = clef-server, ONLY for its torch 2.11.0+cu130 + triton 3.6 that already work on sm_121;
# nothing from clef/ is used. The runtime is copied from semantic-router @ 846e120
# (src/model-runtime) and installed --no-deps so pip cannot touch torch.
FROM clef-server:latest
COPY model-runtime /opt/model-runtime
COPY gb10_unified_memory.patch.py /opt/
RUN python3 /opt/gb10_unified_memory.patch.py /opt/model-runtime/vllm_sr_runtime/accel/gpu.py
RUN pip install --no-cache-dir --no-deps /opt/model-runtime \
 && python3 -c "import vllm_sr_runtime, starlette, uvicorn, prometheus_client, yaml; print('runtime ok')"
ENTRYPOINT ["vllm-sr-runtime", "serve"]
