#!/usr/bin/env python3
"""committee.py — v0 weak-model committee runner for coding tasks.

Implements the setup from "Agentic Systems as Boosting Weak Reasoning
Models" (arXiv 2605.14163), trimmed to the minimum honest experiment we
can run on the DGX Spark:

  1. Read a task directory containing `prompt.md` + one or more
     `test_*.py` files.
  2. Dispatch the prompt to N heterogeneous proposers in parallel
     (each at its own OpenAI-compatible endpoint — Ollama and vLLM both
     work, since both expose /v1/chat/completions).
  3. For each candidate response, drop a `solution.py` into a sandboxed
     temp dir alongside the task's tests, run `pytest`, and record
     pass/fail + timing.
  4. Print a per-proposer summary and the winner.
  5. Persist artifacts (raw response, extracted code, test log) under
     the work dir for later inspection — both winners AND losers,
     because the paper's whole point is that selection failures are
     where the next experiment lives.

WHAT THIS IS NOT
  - Not a benchmark harness. No retries, no grading rubric beyond
    pytest exit code, no statistical aggregation across many tasks.
  - Not a production agent. No tool use, no edit loop, no
    plan-then-code. Single-shot generation, single-shot verification.
  - Not a sampling experiment. One sample per proposer (heterogeneity
    is the variable, not k). Add k-sampling once heterogeneity numbers
    are in.

DESIGN CHOICES THAT MATTER
  - All proposer calls go through the OpenAI-compatible HTTP API via
    urllib (no SDK dependency). Works against vLLM, Ollama, or any
    other server that speaks that protocol.
  - The verifier (pytest) is GROUNDED — it's execution against tests,
    not an LLM judge. That's what makes the paper's 67→76 result
    transfer to this setup. Don't replace it with an LLM critic and
    expect the same gains.
  - Code extraction is intentionally simple: prefer the contents of a
    triple-backtick fenced block, fall back to the raw response. The
    system prompt asks the model to skip fences. Both paths are
    covered because instruction-tuned models often add fences anyway.

TASK DIRECTORY LAYOUT
  task_dir/
    prompt.md          # the prompt sent to each proposer
    test_solution.py   # pytest tests that import from `solution`
    [test_*.py ...]    # additional test files all run together
    [aux_*]            # optional auxiliary files copied into the
                       # sandbox (data fixtures, helpers, etc.)

USAGE
  # Default proposers (Ollama on the Spark, three coders):
  python committee.py --task ./tasks/fibonacci

  # Custom proposer set:
  python committee.py --task ./tasks/fibonacci --proposers my-proposers.json

PROPOSERS FILE FORMAT
  [
    {"name": "qwen-coder-14b",      "endpoint": "http://192.168.1.147:11434/v1", "model": "qwen2.5-coder:14b"},
    {"name": "deepseek-coder-lite", "endpoint": "http://192.168.1.147:11434/v1", "model": "deepseek-coder-v2:16b"},
    {"name": "starcoder2-15b",      "endpoint": "http://192.168.1.147:11434/v1", "model": "starcoder2:15b"}
  ]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_PROPOSERS = [
    {
        "name": "qwen-coder-14b",
        "endpoint": "http://192.168.1.147:11434/v1",
        "model": "qwen2.5-coder:14b",
    },
    {
        "name": "deepseek-coder-lite",
        "endpoint": "http://192.168.1.147:11434/v1",
        "model": "deepseek-coder-v2:16b",
    },
    {
        "name": "starcoder2-15b",
        "endpoint": "http://192.168.1.147:11434/v1",
        "model": "starcoder2:15b",
    },
]

SYSTEM_PROMPT = (
    "You are a Python coding assistant. The user will describe a task. "
    "Reply with ONLY the contents of solution.py — runnable Python, no "
    "explanation, no markdown fences. The tests will import from "
    "`solution` (e.g. `from solution import fibonacci`)."
)

REQUEST_TIMEOUT = 180  # seconds per HTTP call; reasoning models need headroom
PYTEST_TIMEOUT = 60    # seconds per candidate's test run


def call_proposer(
    proposer: dict,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint. Return assistant text."""
    body = json.dumps({
        "model": proposer["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{proposer['endpoint'].rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"].get("content") or ""


def extract_code(text: str) -> str:
    """Prefer the LAST ```python ... ``` block; fall back to whole text.

    Last-block (not first) handles models that emit explanatory code
    earlier and the real answer at the end. If no fence is present,
    assume the model followed the system prompt and the whole response
    is runnable Python.
    """
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    return text.strip()


def run_candidate(
    proposer_name: str,
    code: str,
    task_dir: pathlib.Path,
    work_root: pathlib.Path,
) -> tuple[bool, float, str]:
    """Drop the solution into a sandbox with the tests, run pytest. Return (passed, elapsed_s, log)."""
    sandbox = work_root / proposer_name
    sandbox.mkdir(parents=True, exist_ok=True)
    for tf in task_dir.glob("test_*.py"):
        shutil.copy(tf, sandbox / tf.name)
    for aux in task_dir.glob("aux_*"):
        if aux.is_file():
            shutil.copy(aux, sandbox / aux.name)
    (sandbox / "solution.py").write_text(code)
    t0 = time.time()
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "-q", "--tb=short"],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT,
        )
        elapsed = time.time() - t0
        log = (result.stdout or "") + (result.stderr or "")
        return (result.returncode == 0, elapsed, log)
    except subprocess.TimeoutExpired:
        return (False, float(PYTEST_TIMEOUT), f"(pytest timeout after {PYTEST_TIMEOUT}s)")
    except FileNotFoundError as e:
        return (False, time.time() - t0, f"(missing executable: {e})")


def evaluate(
    proposer: dict,
    prompt: str,
    task_dir: pathlib.Path,
    work_root: pathlib.Path,
    max_tokens: int,
    temperature: float,
) -> dict:
    """End-to-end: call proposer, extract code, run tests. Returns a result dict."""
    name = proposer["name"]
    t0 = time.time()
    try:
        text = call_proposer(proposer, prompt, max_tokens, temperature)
        gen_time = time.time() - t0
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {
            "proposer": name,
            "model": proposer["model"],
            "gen_time": time.time() - t0,
            "gen_error": f"{type(e).__name__}: {e}",
            "raw_response": "",
            "code": "",
            "passed": False,
            "test_time": 0.0,
            "test_log": "",
        }
    code = extract_code(text)
    passed, test_time, test_log = run_candidate(name, code, task_dir, work_root)
    return {
        "proposer": name,
        "model": proposer["model"],
        "gen_time": gen_time,
        "raw_response": text,
        "code": code,
        "passed": passed,
        "test_time": test_time,
        "test_log": test_log,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a coding task across a committee of weak proposers; pytest picks the winner.",
    )
    ap.add_argument("--task", required=True, help="Task directory containing prompt.md + test_*.py")
    ap.add_argument("--proposers", help="JSON file with proposer list (default: built-in)")
    ap.add_argument("--work-dir", help="Where to put sandboxed candidate trees (default: a fresh tempdir)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="Per-proposer max_tokens (default: 2048)")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    args = ap.parse_args()

    task_dir = pathlib.Path(args.task).resolve()
    if not task_dir.is_dir():
        print(f"error: task dir not found: {task_dir}", file=sys.stderr)
        return 2
    prompt_path = task_dir / "prompt.md"
    if not prompt_path.exists():
        print(f"error: missing {prompt_path}", file=sys.stderr)
        return 2
    prompt = prompt_path.read_text()
    tests = sorted(task_dir.glob("test_*.py"))
    if not tests:
        print(f"error: no test_*.py in {task_dir}", file=sys.stderr)
        return 2

    if args.proposers:
        proposers = json.loads(pathlib.Path(args.proposers).read_text())
    else:
        proposers = DEFAULT_PROPOSERS
    if not proposers:
        print("error: empty proposer list", file=sys.stderr)
        return 2

    if args.work_dir:
        work_dir = pathlib.Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = pathlib.Path(tempfile.mkdtemp(prefix="committee-"))

    print("=== committee.py ===")
    print(f"  task:      {task_dir.name}")
    print(f"  tests:     {[t.name for t in tests]}")
    print(f"  proposers: {len(proposers)}")
    for p in proposers:
        print(f"    - {p['name']:25s} {p['model']:30s} @ {p['endpoint']}")
    print(f"  work_dir:  {work_dir}")
    print(f"  T={args.temperature}, max_tokens={args.max_tokens}")
    print()

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(proposers)) as pool:
        futures = {
            pool.submit(evaluate, p, prompt, task_dir, work_dir, args.max_tokens, args.temperature): p
            for p in proposers
        }
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            mark = "✓" if r["passed"] else "✗"
            note = r.get("gen_error") or ("pass" if r["passed"] else "fail")
            print(
                f"  [{mark}] {r['proposer']:25s} "
                f"gen={r['gen_time']:6.1f}s  test={r['test_time']:5.1f}s  {note}"
            )

    print()
    winners = [r for r in results if r["passed"]]
    losers = [r for r in results if not r["passed"]]
    print(f"=== verdict: {len(winners)}/{len(results)} passed ===")
    if winners:
        winners.sort(key=lambda r: r["gen_time"])
        w = winners[0]
        print(f"  winner: {w['proposer']} ({w['model']}) — gen {w['gen_time']:.1f}s, test {w['test_time']:.1f}s")
        if len(winners) > 1:
            for w2 in winners[1:]:
                print(f"  also-passed: {w2['proposer']}")
    for r in losers:
        last_line = ""
        if r.get("gen_error"):
            last_line = r["gen_error"]
        elif r.get("test_log"):
            for line in reversed(r["test_log"].splitlines()):
                if line.strip():
                    last_line = line.strip()
                    break
        print(f"  loser:  {r['proposer']:25s}  →  {last_line}")

    # Persist artifacts for later inspection. Paper's point: both winners
    # and losers are signal — keep the losers' code + test log too.
    for r in results:
        out = work_dir / r["proposer"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "raw_response.txt").write_text(r.get("raw_response") or "")
        (out / "extracted_code.py").write_text(r.get("code") or "")
        (out / "test_log.txt").write_text(r.get("test_log") or "")
    summary = [
        {k: v for k, v in r.items() if k not in ("raw_response", "code", "test_log")}
        for r in results
    ]
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print()
    print(f"artifacts: {work_dir}")
    return 0 if winners else 1


if __name__ == "__main__":
    sys.exit(main())
