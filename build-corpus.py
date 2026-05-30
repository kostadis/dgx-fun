#!/usr/bin/env python3
"""Build a drawer-shaped corpus.txt from a tree of markdown — runs LOCALLY,
then scp the result to the Spark for turbovec-recall-ab.py.

The recall A/B is only honest if the corpus matches how mempalace actually
stores text: paragraph-ish chunks of a few hundred chars, not one-line-per-doc.
This walks a directory, merges paragraphs up to a target size, strips the
loudest markdown noise, dedups, and emits ONE chunk per line (the format the
harness expects — internal newlines collapsed to spaces).

Usage:
    python build-corpus.py ~/src/campaigns-test -o corpus.txt
    python build-corpus.py ~/src/campaigns-test --target 800 --min 200 --max 40000
    scp corpus.txt spark:~/
"""

import argparse
import os
import re
import sys

# Lines that are pure markdown structure, not prose worth embedding.
_SKIP_LINE = re.compile(
    r"^\s*(?:#{1,6}\s|[-*+]\s*$|\|.*\|\s*$|`{3,}|<!--|-{3,}\s*$|={3,}\s*$|\[.*\]:\s)"
)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")          # image embeds
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")        # [text](url) -> text
_WS = re.compile(r"\s+")


def clean(text):
    text = _IMG.sub("", text)
    text = _LINK.sub(r"\1", text)
    return _WS.sub(" ", text).strip()


def paragraphs(raw):
    """Yield blank-line-separated blocks, dropping pure-structure lines."""
    block = []
    for line in raw.splitlines():
        if line.strip() == "":
            if block:
                yield " ".join(block)
                block = []
            continue
        if _SKIP_LINE.match(line):
            # A heading/table/fence ends the current prose block.
            if block:
                yield " ".join(block)
                block = []
            continue
        block.append(line.strip())
    if block:
        yield " ".join(block)


def chunk_file(path, target, maxlen):
    """Merge paragraphs into ~target-char chunks, hard-capped at maxlen."""
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return
    buf = ""
    for para in paragraphs(raw):
        para = clean(para)
        if not para:
            continue
        if not buf:
            buf = para
        elif len(buf) + 1 + len(para) <= target:
            buf += " " + para
        else:
            yield buf[:maxlen]
            buf = para
    if buf:
        yield buf[:maxlen]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory to walk for .md files")
    ap.add_argument("-o", "--out", default="corpus.txt")
    ap.add_argument("--target", type=int, default=800, help="target chunk size in chars")
    ap.add_argument("--min", type=int, default=200, help="drop chunks shorter than this")
    ap.add_argument("--max", type=int, default=40000, help="hard char cap per chunk")
    ap.add_argument("--ext", default=".md", help="comma list of extensions")
    ap.add_argument("--limit", type=int, default=0, help="stop after N chunks (0 = all)")
    args = ap.parse_args()

    exts = tuple(e if e.startswith(".") else "." + e for e in args.ext.split(","))
    files = []
    for dirpath, _, names in os.walk(args.root):
        for nm in names:
            if nm.endswith(exts):
                files.append(os.path.join(dirpath, nm))
    files.sort()  # deterministic
    if not files:
        sys.exit(f"no {exts} files under {args.root!r}")

    seen, kept, short, dropped_dup = set(), [], 0, 0
    for fp in files:
        for ch in chunk_file(fp, args.target, args.max):
            if len(ch) < args.min:
                short += 1
                continue
            if ch in seen:
                dropped_dup += 1
                continue
            seen.add(ch)
            kept.append(ch)
            if args.limit and len(kept) >= args.limit:
                break
        if args.limit and len(kept) >= args.limit:
            break

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept) + "\n")

    lens = [len(c) for c in kept]
    avg = sum(lens) // len(lens) if lens else 0
    print(f"files scanned : {len(files)}")
    print(f"chunks kept   : {len(kept)}")
    print(f"  dropped <{args.min}c : {short}")
    print(f"  dropped dup   : {dropped_dup}")
    print(f"chunk chars   : min {min(lens) if lens else 0}  avg {avg}  max {max(lens) if lens else 0}")
    print(f"wrote         : {args.out}")


if __name__ == "__main__":
    main()
