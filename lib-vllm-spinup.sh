#!/usr/bin/env bash
#
# lib-vllm-spinup.sh — shared helpers sourced by the per-model
# spin-up-vllm-*.sh wrappers. Not directly executable.
#
# The per-model wrappers stay in charge of:
#   - the `docker run` invocation (so flag set is visible per model);
#   - any model-specific pre-setup (plugins, parsers);
#   - any model-specific failure hints (via on_fail_fn arg below).
#
# This library handles the 90% boilerplate that was duplicated across
# spin-up-vllm-gemma.sh / -gemma4-26b-moe.sh / -gemma4-26b-moe-longctx.sh
# / -llama70b.sh / -llama70b-specdecode.sh / -nemotron3-nano-30b.sh:
#
#   * HF_TOKEN extraction from ~/.bashrc under non-interactive SSH
#   * container stop/remove of any prior instance with the same name
#   * pre-launch nvidia-smi snapshot
#   * wait-for-"Application startup complete" loop with heartbeats and
#     early-failure detection
#   * /v1/models + tiny chat-completion smoke tests
#
# Source it from a wrapper with:
#   source "$(dirname "$0")/lib-vllm-spinup.sh"

# ----------------------------------------------------------------------
# vllm_load_hf_token — pick up HF_TOKEN from ~/.bashrc if not in env.
#
# Non-interactive SSH does NOT source .bashrc, so wrappers that rely on
# HF_TOKEN for gated repos need this. We extract just the export line
# rather than sourcing the whole file under `set -euo pipefail`.
# ----------------------------------------------------------------------
vllm_load_hf_token() {
    if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.bashrc" ]; then
        eval "$(grep -E '^[[:space:]]*export[[:space:]]+HF_TOKEN=' "${HOME}/.bashrc" | tail -1)" 2>/dev/null || true
    fi
}

# ----------------------------------------------------------------------
# vllm_stop_container <name>
#
# If a container with this name exists (running or stopped), stop and
# remove it. Errors are swallowed — re-creating the container is what
# matters, not whether stop/rm succeeded against a half-dead instance.
# ----------------------------------------------------------------------
vllm_stop_container() {
    local name="$1"
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "→ stopping existing ${name}..."
        docker stop "${name}" 2>&1 || true
        docker rm "${name}" 2>&1 || true
    fi
}

# ----------------------------------------------------------------------
# vllm_gpu_healthcheck
#
# Print a one-line GPU status before allocating a large slice of VRAM.
# Catches the "previous container didn't release memory" case before
# the new one OOMs at startup.
# ----------------------------------------------------------------------
vllm_gpu_healthcheck() {
    echo "→ GPU status pre-launch:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | head -3
    echo ""
}

# ----------------------------------------------------------------------
# vllm_wait_ready <container> <budget_sec> [error_regex] [on_fail_fn] [tail_lines]
#
# Poll `docker logs <container>` until we see
# "Application startup complete" or until budget_sec elapses. Print a
# heartbeat every 15s with the most recent log line. Bail early if any
# pattern in error_regex shows up.
#
# error_regex (optional) — extended regex matched against the full log
#                          stream. Default covers the common modes:
#                          tracebacks, CUDA errors, OOM, HF repo 404s,
#                          unrecognised CLI args.
# on_fail_fn  (optional) — name of a shell function to call after the
#                          last-N-lines dump on failure, to print
#                          model-specific hints (e.g. "drop MAX_LEN").
# tail_lines  (optional) — how many log lines to dump on failure;
#                          default 60.
#
# Returns 0 on success. Calls `exit 1` on failure (so wrappers running
# under `set -e` get the expected behaviour).
# ----------------------------------------------------------------------
vllm_wait_ready() {
    local container="$1"
    local budget="$2"
    local error_regex="${3:-Error|CUDA error|out of memory|Traceback|404 Client Error|Repository Not Found|unrecognized arguments}"
    local on_fail_fn="${4:-}"
    local tail_lines="${5:-60}"
    local deadline=$(( $(date +%s) + budget ))

    echo "→ waiting for 'Application startup complete' in container logs..."
    echo "  (budget: ${budget}s)"
    echo ""

    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if docker logs "${container}" 2>&1 | grep -q "Application startup complete"; then
            echo "  ✓ ready"
            return 0
        fi
        if docker logs "${container}" 2>&1 | grep -qE "${error_regex}"; then
            echo ""
            echo "  ✗ container errored during startup. Last ${tail_lines} log lines:"
            docker logs --tail "${tail_lines}" "${container}" 2>&1
            if [ -n "${on_fail_fn}" ] && declare -F "${on_fail_fn}" >/dev/null; then
                echo ""
                "${on_fail_fn}"
            fi
            exit 1
        fi
        sleep 15
        local elapsed=$(( budget - (deadline - $(date +%s)) ))
        local last
        last=$(docker logs --tail 1 "${container}" 2>&1 | tr -d '\r' | cut -c1-100)
        echo "  [${elapsed}s] ${last}"
    done

    echo ""
    echo "  ✗ ${budget}-second startup budget exceeded. Last ${tail_lines} log lines:"
    docker logs --tail "${tail_lines}" "${container}" 2>&1
    if [ -n "${on_fail_fn}" ] && declare -F "${on_fail_fn}" >/dev/null; then
        echo ""
        "${on_fail_fn}"
    fi
    exit 1
}

# ----------------------------------------------------------------------
# vllm_smoke_test <host> <port> <model> [max_tokens]
#
# Hit GET /v1/models and POST /v1/chat/completions with a trivial prompt
# to confirm the container is actually responsive (not just that
# uvicorn started). Reasoning models need a non-trivial max_tokens
# because they burn budget on <think> before producing content —
# wrappers for reasoning models should pass max_tokens=2048+.
#
# Default max_tokens is 10 (fine for non-reasoning models).
# ----------------------------------------------------------------------
vllm_smoke_test() {
    local host="$1"
    local port="$2"
    local model="$3"
    local max_tokens="${4:-10}"

    echo ""
    echo "→ smoke-test: GET /v1/models"
    curl -sS --max-time 5 "http://${host}:${port}/v1/models" | python3 -m json.tool 2>&1 | head -20

    echo ""
    echo "→ smoke-test: tiny chat completion (max_tokens=${max_tokens})"
    printf '%s' "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply only with the word OK.\"}],\"max_tokens\":${max_tokens}}" > /tmp/req_smoke.json
    curl -sS --max-time 120 "http://${host}:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @/tmp/req_smoke.json | python3 -m json.tool 2>&1 | head -40
}
