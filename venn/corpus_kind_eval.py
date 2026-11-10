"""Can a venn-1 family suggest a file's mneme corpus kind from its content alone?

Two 5-fold splits over the rows from corpus_kind_build.py:
  dir   - grouped by directory (a held-out file's siblings are never in training; the honest test)
  file  - plain random by file (siblings in training; compared with a path-only baseline)
A label with < 5 training rows in a fold is left out of that fold's family; its held-out rows
count as wrong. Runs in-process with a private registry/cache; the :8007 server is untouched.
    python3 corpus_kind_eval.py rows.json [--form residual|linear]
"""
import argparse, collections, json, random, tempfile, time
from pathlib import Path
from venn import DecisionService
from venn.cli import download_head

ap = argparse.ArgumentParser(); ap.add_argument("rows"); ap.add_argument("--form", default="residual")
ap.add_argument("--cache", default="venn_corpus_cache.sqlite"); ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
R = json.load(open(a.rows))
DESC = {  # one-line option descriptions; the family learns from the examples, these only seed the option vectors
    "npcs": "A dossier about one non-player character: who they are, history, relationships",
    "summaries": "A session summary: what happened at the table in one chapter of play",
    "party": "A player character's sheet or profile",
    "grounding": "A campaign-wide state document: world state, campaign state, party overview, planning",
    "prep": "GM session prep, notes, handouts, stat blocks, threads and locations for upcoming play",
    "arcs": "An arc or villain strategy document tracking a storyline or antagonist across sessions",
    "lore": "Campaign design canon: master plan, guardrails, design frame, rationale",
    "module": "Published-module background or a module inventory",
    "out": "Pipeline output, tool scratch, logs or drafts that are not campaign reference material",
}
S = DecisionService(model_dir=download_head(), cache_path=Path(a.cache).resolve(),
                    registry_path=Path(tempfile.mkdtemp(prefix="venn-ck-")) / "registry.json")
t0 = time.time(); V = S.embed([r["text"] for r in R]); print(f"embedded {len(R)} files in {time.time()-t0:.0f}s", flush=True)
OV = dict(zip(DESC, S.embed(list(DESC.values())))); IV = S.embed(["Which kind of campaign document is this file?"])[0].tolist()

def fit_predict(tag, tr, te):
    cnt = collections.Counter(R[i]["label"] for i in tr); labs = [l for l in DESC if cnt[l] >= 5]
    opts = [{"label": l, "vector": OV[l].tolist()} for l in labs]
    seen, ex = set(), []
    for i in tr:
        if R[i]["label"] in labs and R[i]["text"] not in seen:
            seen.add(R[i]["text"]); ex.append({"state": V[i].tolist(), "label": R[i]["label"], "group_id": R[i]["group"]})
    S.register_family(tag, ex, opts, IV, seed=0, form=a.form)
    q = {"type": "Choice", "family": tag, "instructions_vec": IV, "options": opts}
    out = {}
    for i in te:
        p = S.decide({"state": {"vector": V[i].tolist()}, "questions": {"k": q}})["answers"]["k"]["probabilities"]
        out[i] = (max(p, key=p.get), max(p.values()))
    return out, [l for l in DESC if l not in labs and any(R[i]["label"] == l for i in te)]

def report(name, pred, extra=""):
    n = len(pred); c = sum(pred[i][0] == R[i]["label"] for i in pred)
    per = collections.defaultdict(lambda: [0, 0])
    for i, (x, _) in pred.items(): per[R[i]["label"]][1] += 1; per[R[i]["label"]][0] += x == R[i]["label"]
    conf = sorted(p for _, p in pred.values()); hi = [i for i in pred if pred[i][1] >= conf[len(conf) // 2]]
    print(f"{name:34s} {c}/{n} = {c/n:.1%} | top-half-confidence acc {sum(pred[i][0]==R[i]['label'] for i in hi)/len(hi):.1%} {extra}")
    print("   per kind:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items(), key=lambda kv: -kv[1][1])})
    print("   top confusions (gold->pred):", collections.Counter((R[i]["label"], x) for i, (x, _) in pred.items() if x != R[i]["label"]).most_common(6))

maj = collections.Counter(r["label"] for r in R).most_common(1)[0]
print(f"== venn-1 family form={a.form}; {len(R)} files, {len({r['group'] for r in R})} dirs; majority class '{maj[0]}' = {maj[1]/len(R):.1%}")
for split in ("dir", "file"):
    if split == "dir":
        gs = sorted({r["group"] for r in R}); random.Random(1).shuffle(gs); fold_of = {g: k % a.folds for k, g in enumerate(gs)}
        fk = lambda i: fold_of[R[i]["group"]]
    else:
        idx = list(range(len(R))); random.Random(1).shuffle(idx); fold_of = {i: k % a.folds for k, i in enumerate(idx)}
        fk = lambda i: fold_of[i]
    pred, unseen = {}, []
    for k in range(a.folds):
        te = [i for i in range(len(R)) if fk(i) == k]; tr = [i for i in range(len(R)) if fk(i) != k]
        p, u = fit_predict(f"ck-{split}-{k}", tr, te); pred.update(p); unseen += u
    report(f"venn  split={split}", pred, f"| kinds unseen in some training fold: {sorted(set(unseen))}")
    if split == "file":   # path-only baseline: majority kind of the file's directory among the training rows
        base = {}
        for k in range(a.folds):
            tr = [i for i in range(len(R)) if fk(i) != k]
            bydir = collections.defaultdict(collections.Counter)
            for i in tr: bydir[R[i]["group"]][R[i]["label"]] += 1
            for i in range(len(R)):
                if fk(i) == k:
                    c = bydir.get(R[i]["group"]); base[i] = ((c.most_common(1)[0][0] if c else maj[0]), 1.0)
        report("path-only baseline split=file", base)
