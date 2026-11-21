"""Capture a text-only Anthropic Messages backend behind the local Cognee shim.

Authentication stays in memory. Captures contain both the original messages
and the actual native API body/response, never HTTP authentication headers.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import urllib.error
import urllib.request
import uuid


def thinking_mode(model):
    # Sonnet 5.5 rejects disabled; between_tools suppresses up-front thinking.
    return "between_tools" if model == "claude-sonnet-5-5" else "disabled"


def native_request(body, model):
    if body.get("stream") or body.get("tools"):
        raise ValueError("This experimental adapter accepts non-streaming text only")
    systems, messages = [], []
    for message in body["messages"]:
        if not isinstance(message.get("content"), str):
            raise ValueError("Expected text message content")
        if message["role"] in ("system", "developer"):
            systems.append(message["content"])
        elif message["role"] in ("user", "assistant"):
            messages.append({"role": message["role"], "content": message["content"]})
        else:
            raise ValueError("Unexpected message role")
    return {"model": model, "system": "\n\n".join(systems), "messages": messages,
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 4096)),
            "thinking": {"type": thinking_mode(model)}}


def compatible_response(value):
    usage = value["usage"]
    # Cache tokens are separate in native accounting. No cache is requested here.
    prompt = sum(usage.get(k, 0) for k in
                 ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    output = usage["output_tokens"]
    reason = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
              "refusal": "content_filter"}.get(value["stop_reason"], value["stop_reason"])
    return {"id": value["id"], "object": "chat.completion", "model": value["model"],
            "created": int(time.time()), "choices": [{"index": 0, "finish_reason": reason,
                "message": {"role": "assistant", "content": "".join(
                    b["text"] for b in value["content"] if b["type"] == "text")}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output,
                      "total_tokens": prompt + output}}


class AnthropicCapture:
    def __init__(self, root, model, key_file):
        self.root, self.model = root, model
        self.thinking_mode = thinking_mode(model)
        self.backend = "https://api.anthropic.com"
        self.phase, self.rows = "startup", []
        self.limit, self.lock = threading.BoundedSemaphore(8), threading.Lock()
        api_key = key_file.expanduser().read_text().strip()
        if not api_key.startswith("sk-ant-") or any(c.isspace() for c in api_key):
            raise ValueError("Expected a raw Anthropic API key in the specified file")
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                started = time.monotonic()
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                row = {"id": uuid.uuid4().hex, "phase": capture.phase, "route": "anthropic",
                       "started_unix": time.time(), "request": body,
                       "forwarded": {**body, "model": model},
                       "native_endpoint": capture.backend + "/v1/messages"}
                status = 502
                value = {"error": {"message": "Anthropic adapter failure"}}
                try:
                    native = native_request(body, model)
                    row["native_request"] = native
                    with capture.limit:
                        row["proxy_queue_seconds"] = time.monotonic() - started
                        upstream = time.monotonic()
                        request = urllib.request.Request(row["native_endpoint"],
                            data=json.dumps(native).encode(), headers={
                                "Content-Type": "application/json", "x-api-key": api_key,
                                "anthropic-version": "2023-06-01"})
                        try:
                            with urllib.request.urlopen(request, timeout=300) as response:
                                raw, status = response.read(), response.status
                        except urllib.error.HTTPError as error:
                            raw, status = error.read(), error.code
                        row["upstream_seconds"] = time.monotonic() - upstream
                    row["native_response"] = json.loads(raw)
                    value = (compatible_response(row["native_response"]) if status == 200
                             else row["native_response"])
                except Exception as error:
                    status = 502
                    row["error"] = type(error).__name__
                row.update(seconds=time.monotonic() - started, status=status, response=value)
                path = root / "calls" / (row["id"] + ".json")
                path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
                with capture.lock:
                    capture.rows.append(row)
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
