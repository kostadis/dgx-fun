"""Build the corpus-kind dataset for a mneme campaign: one row per declared file (label = its kind,
from `mneme corpus toc`) plus up to --per-ruling files under each GM ruled-out path (label = out).
Text is the file's first --chars characters (about what the reader sees).
    python3 corpus_kind_build.py <campaign-dir> <out.json>
"""
import argparse, json, random, subprocess
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("campaign_dir", type=Path); ap.add_argument("out")
ap.add_argument("--per-ruling", type=int, default=15); ap.add_argument("--chars", type=int, default=8000)
a = ap.parse_args(); C = a.campaign_dir.resolve()
import yaml
name = yaml.safe_load(open(C / ".mneme/corpus.yaml"))["campaign"]
toc = json.loads(subprocess.run(["mneme", "corpus", "toc", name, "--dir", str(C), "--json"], capture_output=True, text=True, check=True).stdout)
assert toc.get("valid") is not False
rows, seen = [], set()
def add(path, label, group):
    p = C / path
    if path in seen or not p.is_file(): return
    t = p.read_text(errors="replace").strip()[:a.chars]
    if len(t) < 40: return
    seen.add(path); rows.append(dict(path=path, label=label, group=group, text=t))
for kind, body in toc["kinds"].items():
    for f in body["files"]: add(f["path"], kind, str(Path(f["path"]).parent))
rnd = random.Random(0)
for r in yaml.safe_load(open(C / ".mneme/corpus.rulings.yaml"))["excluded"]:
    rp = str(r["path"]); base = C / rp
    fs = sorted(base.rglob("*")) if rp.endswith("/") and base.is_dir() else [base]
    fs = [f for f in fs if f.is_file() and f.suffix in (".md", ".txt")]
    rnd.shuffle(fs)
    for f in fs[:a.per_ruling]: add(str(f.relative_to(C)), "out", "out:" + rp)
json.dump(rows, open(a.out, "w"))
import collections
print(len(rows), collections.Counter(r["label"] for r in rows), "groups", len({r["group"] for r in rows}))
