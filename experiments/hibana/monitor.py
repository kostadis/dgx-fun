"""Sample global unified memory; GPU allocations need not appear in process RSS."""
import json
import time

while True:
    data = {}
    for line in open("/proc/meminfo"):
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "Dirty"):
            data[key + "_kib"] = int(value.split()[0])
    print(json.dumps({"unix": time.time(), **data}), flush=True)
    time.sleep(2)
