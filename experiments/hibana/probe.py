"""Exercise malformed requests and a disconnected stream on the shadow API."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:30004")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    def get(path):
        with urllib.request.urlopen(a.base + path, timeout=10) as response:
            return json.load(response)

    result = {"started_unix": time.time(), "before": get("/get_server_info"), "invalid_requests": []}
    model = get("/v1/models")["data"][0]["id"]
    for change in ({"messages": []}, {"temperature": -1}, {"max_tokens": 0}):
        body = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, **change}
        req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                row = {"change": change, "status": response.status, "body": response.read().decode()}
        except urllib.error.HTTPError as e:
            row = {"change": change, "status": e.code, "body": e.read().decode()}
        result["invalid_requests"].append(row)
    body = {"model": model, "messages": [{"role": "user", "content": "Write a very long numbered list of 1000 reasons to maintain software tests."}],
            "max_tokens": 8192, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if line.startswith(b"data:"):
                event = json.loads(line[5:])
                if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                    result["disconnected_after"] = event
                    break
    for _ in range(20):
        result["after"] = get("/get_server_info")
        if result["after"]["scheduler"]["cancelled"] > result["before"]["scheduler"]["cancelled"]:
            break
        time.sleep(.5)
    result["health"] = get("/health")
    result["passed"] = all(400 <= x["status"] < 500 for x in result["invalid_requests"]) and result["after"]["scheduler"]["cancelled"] > result["before"]["scheduler"]["cancelled"] and result["health"]["status"] == "ok"
    result["finished_unix"] = time.time()
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"passed": result["passed"], "invalid_statuses": [x["status"] for x in result["invalid_requests"]],
                      "cancelled_before": result["before"]["scheduler"]["cancelled"], "cancelled_after": result["after"]["scheduler"]["cancelled"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
