"""Bounded readiness polling with startup-time evidence for a transient unit."""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--unit", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--crash-time", type=Path)
a = p.parse_args()
deadline = time.monotonic() + 600
result = {"passed": False, "unit": a.unit, "poll_interval_seconds": 2}
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:30004/health", timeout=3) as response:
            result["health"] = json.load(response)
        if result["health"]["status"] == "ok":
            result["passed"] = True
            result["first_ready_observed_unix"] = time.time()
            start = int(subprocess.check_output(["systemctl", "--user", "show", a.unit, "--value", "--property=ActiveEnterTimestampMonotonic"]).strip()) / 1e6
            result["seconds_since_unit_start"] = time.monotonic() - start
            if a.crash_time:
                result["seconds_since_crash"] = time.time() - float(a.crash_time.read_text())
            break
    except Exception as e:
        result["last_poll_error"] = repr(e)
    time.sleep(2)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
raise SystemExit(0 if result["passed"] else 1)
