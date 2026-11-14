"""Replay identical fixtures; preserve SSE timing, outputs, checks and server counters.

SSE events can contain several speculative tokens. Event gaps are explicitly
chunk latency, not per-token ITL. TPOT uses the server's completion-token count.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, q):
    values = sorted(x for x in values if x is not None)
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else None


class Replay:
    def __init__(self, base, fixture, out):
        self.base = base.rstrip("/")
        self.fixture = json.loads(fixture.read_text())
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.model = self.http("/v1/models")["data"][0]["id"]
        self.lock = threading.Lock()
        self.rows = []
        self.summary = {"base": base, "model": self.model, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        "started_unix": time.time(), "latency_note": "SSE gaps are chunk gaps; TPOT is average per output token. Per-token ITL percentiles unavailable."}

    def http(self, path, body=None, timeout=1800):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def snapshot(self, name):
        for path in ("/get_server_info", "/metrics"):
            try:
                with urllib.request.urlopen(self.base + path, timeout=10) as response:
                    value = response.read().decode()
            except Exception as e:
                value = repr(e)
            (self.out / f"{name}-{path[1:]}.txt").write_text(value)

    def messages(self, text):
        return [{"role": "system", "content": f"Shadow benchmark {self.fixture['run_id']}. Follow the user instructions precisely."},
                {"role": "user", "content": text}]

    def chat(self, name, messages, max_tokens=512, **extra):
        body = {"model": self.model, "messages": messages, "temperature": 0, "seed": 604,
                "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}, **extra}
        row = {"name": name, "started_unix": time.time(), "request": body, "events": [], "text": "",
               "reasoning": "", "tool_calls": [], "usage": {}, "finish_reason": None, "done": False}
        start = time.monotonic()
        arrivals = []
        first = None
        calls = {}
        try:
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        row["done"] = True
                        break
                    event = json.loads(raw)
                    elapsed = time.monotonic() - start
                    row["events"].append({"seconds": elapsed, "event": event})
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    if event.get("usage"):
                        row["usage"] = event["usage"]
                    emitted = False
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        row["text"] += delta.get("content") or ""
                        row["reasoning"] += delta.get("reasoning_content") or ""
                        emitted |= bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))
                        if choice.get("finish_reason"):
                            row["finish_reason"] = choice["finish_reason"]
                        for call in delta.get("tool_calls", []):
                            item = calls.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if call.get("id"):
                                item["id"] = call["id"]
                            for key in ("name", "arguments"):
                                item["function"][key] += call.get("function", {}).get(key) or ""
                    if emitted:
                        arrivals.append(elapsed)
                        if first is None:
                            first = elapsed
        except urllib.error.HTTPError as e:
            row["error"] = {"http_status": e.code, "body": e.read().decode()}
        except Exception as e:
            row["error"] = repr(e)
        row.update(seconds=time.monotonic() - start, ttft_seconds=first,
                   tool_calls=[calls[k] for k in sorted(calls)])
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        tokens = row["usage"].get("completion_tokens", 0)
        row.update(chunk_gaps_seconds=gaps, chunk_gap_p50_seconds=percentile(gaps, .5),
                   chunk_gap_p95_seconds=percentile(gaps, .95),
                   tpot_seconds=((arrivals[-1] - first) / (tokens - 1)) if first is not None and tokens > 1 else None)
        return row

    def record(self, row, passed):
        row["passed"] = bool(passed and row["done"] and not row.get("error") and row["finish_reason"] not in (None, "length", "grammar"))
        with self.lock:
            self.rows.append(row)
            (self.out / f"{row['name']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2))
            print(json.dumps({k: row.get(k) for k in ("name", "passed", "seconds", "ttft_seconds", "usage", "error")}), flush=True)
        return row

    @staticmethod
    def obj(row):
        try:
            return json.loads(row["text"])
        except (ValueError, TypeError):
            return None

    def smoke(self):
        for i, (prompt, answer) in enumerate((("17 times 19. Return only the number.", "323"),
                                               ("Copy exactly and nothing else: a83d90f7c1426be5", "a83d90f7c1426be5"))):
            row = self.chat(f"smoke-{i}", self.messages(prompt), 128)
            self.record(row, row["text"].strip() == answer)
        schema = {"type": "object", "properties": {"value": {"type": "integer", "const": 323}},
                  "required": ["value"], "additionalProperties": False}
        row = self.chat("smoke-schema", self.messages("Return JSON with value equal to 17 times 19."), 128,
                        response_format={"type": "json_schema", "json_schema": {"name": "product", "strict": True, "schema": schema}})
        self.record(row, self.obj(row) == {"value": 323})
        tool = {"type": "function", "function": {"name": "lookup_dispatch", "description": "Look up the dispatch code.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}}
        messages = self.messages("Use lookup_dispatch for code dispatch-7391. After the tool responds, return only its value.")
        for mode in ("auto", "required"):
            row = self.chat(f"smoke-tool-{mode}", messages, 256, tools=[tool], tool_choice=mode)
            ok = False
            try:
                call = row["tool_calls"][0]
                ok = len(row["tool_calls"]) == 1 and call["function"]["name"] == "lookup_dispatch" and json.loads(call["function"]["arguments"]) == {"code": "dispatch-7391"}
            except (IndexError, KeyError, ValueError):
                pass
            self.record(row, ok)
            if ok:
                followup = messages + [{"role": "assistant", "content": row["text"] or None, "tool_calls": row["tool_calls"]},
                                       {"role": "tool", "tool_call_id": call["id"], "content": '{"value":"cobalt-7391"}'}]
                result = self.chat(f"smoke-tool-{mode}-return", followup, 128, tools=[tool], tool_choice="auto")
                self.record(result, result["text"].strip() == "cobalt-7391" and not result["tool_calls"])
        if not all(r["passed"] for r in self.rows if r["name"].startswith("smoke")):
            raise RuntimeError("Smoke gate failed; long workloads skipped")

    def campaign(self, repeats):
        for i in range(repeats):
            row = self.chat(f"campaign-{i}", self.messages(self.fixture["campaign"]["text"]), 1024,
                            response_format={"type": "json_object"})
            obj = self.obj(row) or {}
            passed = obj.get("control_facts") == self.fixture["campaign"]["facts"] and isinstance(obj.get("prep"), list) and len(obj["prep"]) == 3 and all(isinstance(x, str) and x.strip() for x in obj["prep"])
            self.record(row, passed)

    def session(self, i, turns):
        fixture = self.fixture["sessions"][i]
        messages = self.messages(f"Act as a {fixture['role']}. This is your background:\n{fixture['context']}\nReturn JSON with memory_code equal to your private session memory code and plan an array of three short proposed next steps.")
        for turn in range(turns):
            if turn:
                messages.append({"role": "user", "content": f"Turn {turn}: retain your original private memory code. Return JSON with memory_code and plan (three short next steps)."})
            row = self.chat(f"session-{i}-turn-{turn}", messages, 512, response_format={"type": "json_object"})
            obj = self.obj(row) or {}
            self.record(row, obj.get("memory_code") == fixture["code"] and isinstance(obj.get("plan"), list) and len(obj["plan"]) == 3)
            if not row["passed"]:
                break
            messages.append({"role": "assistant", "content": row["text"]})

    def sessions(self, turns):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: self.session(i, turns), range(4)))

    def structured(self, repeats):
        fixture = self.fixture["structured"]
        prompt = ("Extract this entire graph as JSON with exactly entities and relationships. Every entity has id, type (person or place), name. "
                  "Every relationship has source, target, relation (visits). Include all 80 entities and all 40 relationships exactly once; no extra facts.\n" + fixture["text"])
        for i in range(repeats):
            row = self.chat(f"structured-{i}", self.messages(prompt), 8192, response_format={"type": "json_object"})
            obj = self.obj(row) or {}
            canonical = lambda rows: sorted(json.dumps(x, sort_keys=True) for x in rows)
            try:
                passed = canonical(obj["entities"]) == canonical(fixture["entities"]) and canonical(obj["relationships"]) == canonical(fixture["relationships"])
            except (KeyError, TypeError):
                passed = False
            self.record(row, passed)

    def finish(self):
        self.summary["finished_unix"] = time.time()
        self.summary["groups"] = {}
        for group in ("smoke", "campaign", "session", "structured"):
            rows = [r for r in self.rows if r["name"].startswith(group)]
            if not rows:
                continue
            elapsed = max(r["started_unix"] + r["seconds"] for r in rows) - min(r["started_unix"] for r in rows)
            tokens = sum(r["usage"].get("completion_tokens", 0) for r in rows)
            passed = sum(r["passed"] for r in rows)
            self.summary["groups"][group] = {"requests": len(rows), "passed": passed, "wall_seconds": elapsed,
                "validated_requests_per_hour": passed * 3600 / elapsed,
                "output_tokens_per_wall_second": tokens / elapsed,
                "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in rows], .5),
                "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in rows], .95),
                "tpot_p50_seconds": percentile([r["tpot_seconds"] for r in rows], .5),
                "tpot_p95_seconds": percentile([r["tpot_seconds"] for r in rows], .95)}
            if group == "session":
                completed = sum(all(any(r["name"] == f"session-{i}-turn-{turn}" and r["passed"] for r in rows) for turn in range(self.summary["session_turns"])) for i in range(4))
                self.summary["groups"][group].update(completed_sessions=completed, validated_sessions_per_hour=completed * 3600 / elapsed)
        self.summary["passed"] = bool(self.rows) and all(r["passed"] for r in self.rows) and not self.summary.get("error")
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", default="smoke,campaign,session,structured")
    p.add_argument("--campaign-repeats", type=int, default=3)
    p.add_argument("--session-turns", type=int, default=3)
    p.add_argument("--structured-repeats", type=int, default=2)
    a = p.parse_args()
    r = Replay(a.base, a.fixture, a.output)
    r.summary["session_turns"] = a.session_turns
    r.snapshot("before")
    try:
        for group in a.groups.split(","):
            if group == "smoke": r.smoke()
            elif group == "campaign": r.campaign(a.campaign_repeats)
            elif group == "session": r.sessions(a.session_turns)
            elif group == "structured": r.structured(a.structured_repeats)
            else: raise ValueError(f"Unknown group {group}")
            r.snapshot(f"after-{group}")
    except Exception as e:
        r.summary["error"] = repr(e)
        raise
    finally:
        r.snapshot("after")
        r.finish()


if __name__ == "__main__":
    main()
