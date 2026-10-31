#!/usr/bin/env python3
"""Gemma 4 26B-A4B NVFP4 trial benchmark (stdlib only, run on the box).

Measures both axes (feedback_llm_bench_prefill_vs_decode):
  decode   - single-stream tok/s on a ~1000-token generation, tiny prompt
  concur   - aggregate tok/s at N concurrent streams (article: 271 @ 8)
  prefill  - cold TTFT at ~8K / 32K / 100K tokens, unique nonce per prompt so
             prefix caching cannot fake a cold number; then a warm resend
  tools    - one forced-relevant tool call, checks finish_reason + arguments
  think    - reasoning on vs off, which response field carries the trace

Usage: python3 -I bench.py [base_url] [model] > results.json
"""
import json, sys, time, threading, uuid, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "google/gemma-4-26B-A4B-it"


def post(path, body, stream=False, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat_stream(messages, max_tokens, extra=None):
    """Return (ttft_s, total_s, completion_tokens, prompt_tokens)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if extra:
        body.update(extra)
    t0 = time.time()
    ttft = None
    usage = {}
    chunks = 0  # fallback token count when the server sends no usage
    with post("/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning")
                        or d.get("reasoning_content")):
                    chunks += 1
                    if ttft is None:
                        ttft = time.time() - t0
    return ttft, time.time() - t0, usage.get("completion_tokens") or chunks, \
        usage.get("prompt_tokens", 0)


LONG_OUT = [{"role": "user", "content":
             "Write a detailed 1500-word short story about a lighthouse keeper "
             "who discovers a message in a bottle. Do not stop early."}]


def bench_decode():
    chat_stream([{"role": "user", "content": "Say hi."}], 8)  # warm sampler
    ttft, tot, n, _ = chat_stream(LONG_OUT, 1000, {"ignore_eos": True})
    return {"completion_tokens": n, "ttft_s": round(ttft, 3),
            "total_s": round(tot, 2),
            "decode_tok_s": round((n - 1) / (tot - ttft), 1)}


def bench_concurrency(n):
    out = [None] * n

    def run(i):
        out[i] = chat_stream(LONG_OUT, 500, {"ignore_eos": True})

    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    toks = sum(o[2] for o in out)
    return {"streams": n, "total_tokens": toks, "wall_s": round(wall, 2),
            "aggregate_tok_s": round(toks / wall, 1),
            "per_stream_tok_s": round(toks / wall / n, 1)}


FILLER = ("The archive ledger records shipments of grain, salt, timber and "
          "iron between the river towns, with dates, tallies and the names of "
          "the clerks who signed each entry. ")


def long_prompt(approx_tokens):
    nonce = uuid.uuid4().hex
    body = FILLER * (approx_tokens // 30)
    return [{"role": "user", "content":
             f"[{nonce}]\n{body}\nIn one sentence: what goods are listed?"}]


def bench_prefill(sizes):
    res = []
    for sz in sizes:
        msgs = long_prompt(sz)
        cold = chat_stream(msgs, 16)
        warm = chat_stream(msgs, 16)
        res.append({"target_tokens": sz, "prompt_tokens": cold[3],
                    "cold_ttft_s": round(cold[0], 2),
                    "cold_prefill_tok_s": round(cold[3] / cold[0]),
                    "warm_ttft_s": round(warm[0], 2)})
        print(json.dumps(res[-1]), file=sys.stderr)
    return res


def bench_tools():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    body = {"model": MODEL, "temperature": 0, "max_tokens": 256,
            "tools": tools, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user",
                          "content": "What's the weather in Lisbon right now?"}]}
    t0 = time.time()
    r = json.load(post("/chat/completions", body))
    ch = r["choices"][0]
    return {"latency_s": round(time.time() - t0, 2),
            "finish_reason": ch["finish_reason"],
            "tool_calls": ch["message"].get("tool_calls"),
            "content": ch["message"].get("content")}


def bench_think():
    out = {}
    for on in (True, False):
        body = {"model": MODEL, "temperature": 0, "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": on},
                "messages": [{"role": "user", "content":
                              "A bat and a ball cost $1.10 total. The bat costs "
                              "$1.00 more than the ball. How much is the ball? "
                              "Answer with just the amount."}]}
        t0 = time.time()
        r = json.load(post("/chat/completions", body))
        m = r["choices"][0]["message"]
        out["on" if on else "off"] = {
            "latency_s": round(time.time() - t0, 2),
            "completion_tokens": r["usage"]["completion_tokens"],
            "content": m.get("content"),
            "reasoning_field": "reasoning" if m.get("reasoning") else
            ("reasoning_content" if m.get("reasoning_content") else None),
            "reasoning_chars": len(m.get("reasoning") or
                                   m.get("reasoning_content") or "")}
    return out


if __name__ == "__main__":
    res = {"model": MODEL, "base": BASE, "started": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, fn in [("tools", bench_tools), ("think", bench_think),
                     ("decode", bench_decode),
                     ("concurrency_1", lambda: bench_concurrency(1)),
                     ("concurrency_4", lambda: bench_concurrency(4)),
                     ("concurrency_8", lambda: bench_concurrency(8)),
                     ("prefill", lambda: bench_prefill([8000, 32000, 100000]))]:
        try:
            res[name] = fn()
        except Exception as e:  # keep going; record the failure
            res[name] = {"error": repr(e)}
        print(name, json.dumps(res[name])[:300], file=sys.stderr)
    print(json.dumps(res, indent=2))
