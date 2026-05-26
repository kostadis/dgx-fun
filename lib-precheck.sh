#!/usr/bin/env bash
#
# lib-precheck.sh — shared precheck helpers for any wrapper that
# launches against a vLLM endpoint on the Spark and wants to confirm
# the right model is loaded at sufficient context.
#
# Why this exists separately from lib-vllm-spinup.sh:
#   spinup runs ON the Spark and manages the container. Precheck runs
#   on the CLIENT side (workstation, ssh wrapper, committee orchestrator)
#   and just inspects /v1/models on whatever endpoint we point it at.
#
# Source from a wrapper:
#   source "$(dirname "$0")/lib-precheck.sh"

# ----------------------------------------------------------------------
# precheck_vllm_endpoint <endpoint> <expected_model_id> <min_ctx>
#
# endpoint           e.g. http://192.168.1.147:8001/v1
# expected_model_id  the model id that GET /models should return as the
#                    first served model (matched exactly).
# min_ctx            minimum acceptable max_model_len from the same
#                    response. Pass 0 to skip the context check.
#
# On success: prints "  ✓ <model> @ max_model_len=<ctx>" and returns 0.
# On failure: prints a diagnostic message to stderr and returns 1.
# The caller is responsible for any model-specific "to fix, run this
# spin-up script" hints — keep them in the wrapper, not here.
# ----------------------------------------------------------------------
precheck_vllm_endpoint() {
    local endpoint="$1"
    local expected_model="$2"
    local min_ctx="$3"

    local models_json
    models_json="$(curl -sS --max-time 5 "${endpoint}/models" 2>&1 || true)"

    if ! printf '%s' "${models_json}" | grep -q '"data"'; then
        echo "  ✗ ${endpoint}/models did not respond with a model list." >&2
        echo "    response was:" >&2
        printf '%s\n' "${models_json}" | head -5 >&2
        return 1
    fi

    local served_model
    served_model="$(printf '%s' "${models_json}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
ids = [m.get("id") for m in data.get("data", [])]
print(ids[0] if ids else "")
' 2>/dev/null || true)"

    if [ "${served_model}" != "${expected_model}" ]; then
        echo "  ✗ expected model ${expected_model}, got '${served_model}'." >&2
        return 1
    fi

    if [ "${min_ctx}" -gt 0 ]; then
        local ctx
        ctx="$(printf '%s' "${models_json}" | python3 -c '
import json, sys
data = json.load(sys.stdin)
items = data.get("data", [])
print(items[0].get("max_model_len", 0) if items else 0)
' 2>/dev/null || echo 0)"
        if [ "${ctx}" -lt "${min_ctx}" ]; then
            echo "  ✗ max_model_len=${ctx} is below min_ctx=${min_ctx}." >&2
            return 1
        fi
        echo "  ✓ ${served_model} @ max_model_len=${ctx}"
    else
        echo "  ✓ ${served_model} (context check skipped)"
    fi
    return 0
}
