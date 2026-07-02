#!/usr/bin/env python3
"""batch_status.py — progress report for a batch_convert.py run.

Reads batch_convert's SQLite **state DB** (read-only) and reports honest progress
toward the *deduped* goal (canonical, non-skipped docs), counting only real
adventure JSON on disk — NOT the per-chunk caches under each `<stem>-responses/`
directory. Also shows whether the batch is alive and which docs are in flight.

The state DB is batch_convert's single source of truth — it pulls the
canonical/dedup decision from rpg-lib's API and records it there, so this tool
never contacts rpg-lib (or its DB) directly. WAL + read-only mode let us read a
consistent committed snapshot while a conversion is writing.

It also reports the cumulative chunk count (every `*-parsed.json` the converter
has produced, across all docs — a monotonic running total), and in --watch mode
the per-interval chunk delta + a session chunks/hr, a finer-grained throughput
signal than docs/hr (chunks tick steadily between whole-doc completions).

Usage:
    python3 batch_status.py                 # one-shot report (incl. total chunks)
    python3 batch_status.py --watch 30      # refresh 30s: docs/hr + ETA + chunks/30s
    python3 batch_status.py --fast          # state-only (skip on-disk JSON + chunk scan)
    python3 batch_status.py --state-db X.db
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path

# This tool lives with the DGX throughput experiments but reads the
# pdf-translators batch state DB. Default to that path; override with --state-db.
DEFAULT_STATE_DB = Path(
    "/home/kroussos/src/mytools/pdf-translators/dmsguild-state.db"
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


def chunk_count(pdf_path: str) -> int:
    """JSON chunks the converter has produced so far for this doc — the
    `*-parsed.json` files in its `<stem>-responses/` dir. 0 if not started."""
    p = Path(pdf_path)
    rdir = p.parent / f"{p.stem}-responses"
    if not rdir.is_dir():
        return 0
    return sum(1 for _ in rdir.glob("*-parsed.json"))


def total_chunks_on_disk(root: Path) -> int:
    """Cumulative count of converter output chunks across the whole batch — every
    `*-parsed.json` inside any `*-responses/` dir under `root`, not just in-flight
    docs. batch_convert never deletes these (they back `--reuse-responses`), so
    this is a true *running total* — monotonic across a run. The tree walk is
    slow on network mounts, so callers skip it under `--fast`."""
    if not root or not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*-parsed.json")
               if p.parent.name.endswith("-responses"))


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


def _open_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection — a consistent WAL snapshot, never blocks the writer."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def compute(state_db: Path, check_disk: bool) -> dict:
    conn = _open_ro(state_db)
    try:
        meta = {r["key"]: r["value"]
                for r in conn.execute("SELECT key, value FROM meta")}
        rows = conn.execute("SELECT rel, status, reason FROM docs").fetchall()
        # In-flight docs: the encode phase runs in-process (no converter
        # subprocess to pgrep), so it marks each doc status='running' with its
        # endpoint while converting. Read that here.
        running_rows = conn.execute(
            "SELECT rel, endpoint FROM docs WHERE status='running'").fetchall()
    finally:
        conn.close()

    root = Path(meta.get("root", ""))
    saved_at = int(meta.get("saved_at", 0) or 0)
    flights = [(str(root / r["rel"]), r["endpoint"] or "?") for r in running_rows]

    docs = {r["rel"]: {"status": r["status"], "reason": r["reason"]} for r in rows}
    canonical = {r: v for r, v in docs.items() if v["status"] != "skipped"}
    skipped = {r: v for r, v in docs.items() if v["status"] == "skipped"}

    # Real deliverable = final adventure JSON next to the PDF (never inside a
    # *-responses/ dir). Prefer disk truth; fall back to the state 'done' status.
    def jpath(rel: str) -> Path:
        return (root / rel).with_suffix(".json")

    if check_disk:
        converted = sum(1 for r in canonical if jpath(r).exists())
        stale = sum(1 for r in skipped if jpath(r).exists())
        source = "json on disk"
    else:
        converted = sum(1 for v in canonical.values() if v["status"] == "done")
        stale = 0
        source = "state db status"

    failed = sum(1 for v in canonical.values() if v["status"] == "failed")
    remaining = len(canonical) - converted - failed

    skip_reasons = Counter(v["reason"] or "?" for v in skipped.values())

    # Cumulative chunk total — the fine-grained progress signal between whole-doc
    # completions. Same on-disk cost class as the converted-count, so gate it on
    # check_disk; None under --fast.
    chunks_total = total_chunks_on_disk(root) if check_disk else None

    return {
        "root": root, "saved_at": saved_at,
        "target": len(canonical), "converted": converted, "remaining": remaining,
        "failed": failed, "skipped": len(skipped), "stale": stale,
        "skip_reasons": skip_reasons, "source": source,
        "chunks_total": chunks_total, "flights": flights,
    }


def render(st: dict, verbose: bool) -> None:
    running, pid = driver_running()
    # In-flight docs come from the state DB (status='running'); only trust them
    # while the driver is alive, else they're stale 'running' rows from a crash.
    flights = st.get("flights", []) if running else []
    now = time.time()

    age = _age(now - st["saved_at"]) if st["saved_at"] else "unknown"
    print(f"Conversion progress   (state saved {age})")
    if running:
        print(f"  batch RUNNING (pid {pid}) — {len(flights)} doc(s) in flight (in-process)")
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

    if st.get("chunks_total") is not None:
        print(f"  Chunks produced (cumulative, *-parsed.json on disk): "
              f"{st['chunks_total']}")
        print()

    if flights:
        total_chunks = 0
        by_ep: dict[str, list[tuple[str, int]]] = {}
        for pdf_path, ep in flights:
            n = chunk_count(pdf_path)
            total_chunks += n
            by_ep.setdefault(ep, []).append((os.path.basename(pdf_path), n))
        print(f"  In flight now ({total_chunks} json chunks done across "
              f"{len(flights)} docs):")
        for ep in sorted(by_ep):
            print(f"    {ep}")
            for name, n in by_ep[ep]:
                print(f"      {n:>4} chunks  {name}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-db", "--manifest", dest="state_db", type=Path,
                   default=DEFAULT_STATE_DB,
                   help="batch_convert's SQLite state DB (--manifest is a "
                        "deprecated alias).")
    p.add_argument("--watch", type=float, metavar="SECONDS", default=None,
                   help="Refresh every SECONDS; show docs/hr and ETA from the delta.")
    p.add_argument("--fast", action="store_true",
                   help="State-only; skip stat-ing JSON on disk (faster on slow mounts).")
    p.add_argument("--verbose", action="store_true", help="Break down skip reasons.")
    args = p.parse_args(argv)

    if not args.state_db.exists():
        print(f"error: state DB not found: {args.state_db}")
        return 1

    if args.watch is None:
        render(compute(args.state_db, check_disk=not args.fast), args.verbose)
        return 0

    # Watch mode: sample docs/hr from the converted-count delta, and chunk
    # throughput from the cumulative-chunk delta (per-interval + session avg).
    first = None        # (t, converted) — anchors the docs/hr session rate + ETA
    first_ch = None     # (t, chunks)    — anchors the chunks/hr session rate
    prev_ch = None      # (t, chunks)    — previous tick, for the per-interval delta
    try:
        while True:
            st = compute(args.state_db, check_disk=not args.fast)
            now = time.time()
            os.system("clear")
            render(st, args.verbose)
            if first is None:
                first = (now, st["converted"])
            else:
                dt = (now - first[0]) / 3600.0
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

            # Chunk throughput — finer-grained than docs/hr (chunks tick between
            # whole-doc completions). Skipped under --fast (chunks_total is None).
            ch = st.get("chunks_total")
            if ch is not None:
                if first_ch is None:
                    first_ch = (now, ch)
                    print(f"  chunks: {ch} total   (measuring rate…)")
                else:
                    d_ch = ch - prev_ch[1]
                    d_t = now - prev_ch[0]
                    sess_dt = (now - first_ch[0]) / 3600.0
                    sess_rate = (ch - first_ch[1]) / sess_dt if sess_dt > 0 else 0.0
                    print(f"  chunks: {ch} total   +{d_ch} in last {d_t:.0f}s"
                          f"   ({sess_rate:.0f} chunks/hr session avg)")
                prev_ch = (now, ch)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
