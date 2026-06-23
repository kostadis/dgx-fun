#!/usr/bin/env python3
"""batch_status.py — progress report for a batch_convert.py run.

Reads the batch manifest and reports honest progress toward the *deduped* goal
(canonical, non-skipped docs), counting only real adventure JSON on disk — NOT
the per-chunk caches under each `<stem>-responses/` directory. Also shows
whether the batch is alive and which docs are in flight right now.

Usage:
    python3 batch_status.py                 # one-shot report
    python3 batch_status.py --watch 30      # refresh every 30s, with docs/hr + ETA
    python3 batch_status.py --fast          # manifest-only (skip the on-disk JSON stat)
    python3 batch_status.py --manifest X.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

# This tool lives with the DGX throughput experiments but reads the
# pdf-translators batch manifest. Default to that path; override with --manifest.
DEFAULT_MANIFEST = Path(
    "/home/kroussos/src/mytools/pdf-translators/dmsguild-manifest.json"
)


def _pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-af", pattern],
                             capture_output=True, text=True)
    except FileNotFoundError:
        return []
    lines = []
    for ln in out.stdout.splitlines():
        if "pgrep" in ln or "batch_status.py" in ln:
            continue
        lines.append(ln)
    return lines


def driver_running() -> tuple[bool, str | None]:
    lines = _pgrep(r"batch_convert\.py")
    if not lines:
        return False, None
    pid = lines[0].split(None, 1)[0]
    return True, pid


def inflight_docs() -> list[tuple[str, str]]:
    """(pdf_basename, endpoint) for each live converter subprocess."""
    docs = []
    for ln in _pgrep(r"pdf_to_5etools_v2\.py"):
        mp = re.search(r"pdf_to_5etools_v2\.py\s+(.+?)\s+--provider", ln)
        me = re.search(r"--endpoint\s+(\S+)", ln)
        if mp:
            docs.append((os.path.basename(mp.group(1)), me.group(1) if me else "?"))
    return docs


def _bar(frac: float, width: int = 24) -> str:
    fill = int(round(frac * width))
    return "▓" * fill + "░" * (width - fill)


def _age(seconds: float) -> str:
    s = int(seconds)
    if s < 90:
        return f"{s}s ago"
    if s < 5400:
        return f"{s // 60}m ago"
    return f"{s / 3600:.1f}h ago"


def compute(manifest_path: Path, check_disk: bool) -> dict:
    m = json.loads(manifest_path.read_text())
    root = Path(m["root"])
    docs = m["docs"]
    saved_at = m.get("saved_at", 0)

    canonical = {r: v for r, v in docs.items() if v.get("status") != "skipped"}
    skipped = {r: v for r, v in docs.items() if v.get("status") == "skipped"}

    # Real deliverable = final adventure JSON next to the PDF (never inside a
    # *-responses/ dir). Prefer disk truth; fall back to manifest 'done' status.
    def jpath(rel: str) -> Path:
        return (root / rel).with_suffix(".json")

    if check_disk:
        converted = sum(1 for r in canonical if jpath(r).exists())
        stale = sum(1 for r in skipped if jpath(r).exists())
        source = "json on disk"
    else:
        converted = sum(1 for v in canonical.values() if v.get("status") == "done")
        stale = 0
        source = "manifest status"

    failed = sum(1 for v in canonical.values() if v.get("status") == "failed")
    remaining = len(canonical) - converted - failed

    skip_reasons = Counter(v.get("reason", "?") for v in skipped.values())

    return {
        "root": root, "saved_at": saved_at,
        "target": len(canonical), "converted": converted, "remaining": remaining,
        "failed": failed, "skipped": len(skipped), "stale": stale,
        "skip_reasons": skip_reasons, "source": source,
    }


def render(st: dict, verbose: bool) -> None:
    running, pid = driver_running()
    flights = inflight_docs()
    now = time.time()

    age = _age(now - st["saved_at"]) if st["saved_at"] else "unknown"
    print(f"Conversion progress   (manifest saved {age})")
    if running:
        print(f"  batch RUNNING (pid {pid}) — {len(flights)} converter subprocess(es) in flight")
    else:
        print(f"  batch NOT running")
    print()

    tgt = st["target"] or 1
    frac = st["converted"] / tgt
    print(f"  Toward the deduped goal (canonical docs, source: {st['source']}):")
    print(f"    target      {st['target']:>5}")
    print(f"    converted   {st['converted']:>5}   {_bar(frac)}  {100*frac:.1f}%")
    print(f"    remaining   {st['remaining']:>5}")
    print(f"    failed      {st['failed']:>5}   (re-run to retry; --reuse-responses keeps partial work)")
    print()

    print(f"  Skipped as non-canonical: {st['skipped']}"
          + (f"   (stale JSON on disk: {st['stale']})" if st["stale"] else ""))
    if verbose and st["skip_reasons"]:
        for reason, n in sorted(st["skip_reasons"].items()):
            print(f"      {reason}: {n}")
    print()

    if flights:
        print("  In flight now:")
        by_ep: dict[str, list[str]] = {}
        for name, ep in flights:
            by_ep.setdefault(ep, []).append(name)
        for ep in sorted(by_ep):
            print(f"    {ep}")
            for name in by_ep[ep]:
                print(f"      - {name}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--watch", type=float, metavar="SECONDS", default=None,
                   help="Refresh every SECONDS; show docs/hr and ETA from the delta.")
    p.add_argument("--fast", action="store_true",
                   help="Manifest-only; skip stat-ing JSON on disk (faster on slow mounts).")
    p.add_argument("--verbose", action="store_true", help="Break down skip reasons.")
    args = p.parse_args(argv)

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}")
        return 1

    if args.watch is None:
        render(compute(args.manifest, check_disk=not args.fast), args.verbose)
        return 0

    # Watch mode: sample the rate from the converted-count delta.
    first = None
    t0 = time.time()
    try:
        while True:
            st = compute(args.manifest, check_disk=not args.fast)
            os.system("clear")
            render(st, args.verbose)
            if first is None:
                first = (time.time(), st["converted"])
            else:
                dt = (time.time() - first[0]) / 3600.0
                dn = st["converted"] - first[1]
                if dt > 0 and dn > 0:
                    rate = dn / dt
                    eta_h = st["remaining"] / rate if rate else float("inf")
                    print()
                    print(f"  rate: {rate:.1f} docs/hr   ETA for {st['remaining']} "
                          f"remaining: {eta_h:.1f}h   (sampled over {dt*60:.0f} min)")
                else:
                    print()
                    print(f"  rate: measuring… (no completion yet in this window)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
