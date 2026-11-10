"""E6 for Venn: train a venn-1 family head on the GM's rulings; evaluate by held-out group.

Same items, folds (groups shuffled with Random(1), i::folds), thresholds and metrics as
train_head.py, so the numbers sit beside tuned Nox. Venn picks weight decay / epochs (or C)
itself by grouped CV inside each training slice. Runs in-process with a private registry and
vector cache; the served :8007 instance is not touched.
    python3 venn_family.py e6_triage_items.json --task triage --folds 3 [--form residual|linear] [--balanced]
"""
import argparse, collections, json, random, tempfile, time
from pathlib import Path
from venn import DecisionService
from venn.cli import download_head

ap = argparse.ArgumentParser()
ap.add_argument("items"); ap.add_argument("--task", required=True); ap.add_argument("--folds", type=int, default=3)
ap.add_argument("--form", default="residual"); ap.add_argument("--balanced", action="store_true")
ap.add_argument("--cache", default="venn_e6_cache.sqlite"); ap.add_argument("--max-epochs", type=int)
a = ap.parse_args()
items = json.load(open(a.items))
crit = items[0]["questions"]["kind"]["criteria"]; instr = items[0]["questions"]["kind"]["instructions"]
if a.task == "triage":   # the metric is P(campaign); the three gold labels are what the GM ruled
    label = lambda r: r["label"]
    LABELS = {"leave": crit["filler"] + " / " + crit["rules"], "chatter": crit["realworld"], "name": crit["campaign"]}
    group = lambda r: r["session"]
else:                    # event/thread/date are never gold, so the head only knows the five entity types
    label = lambda r: r["gold"]
    LABELS = {k: crit[k] for k in ("npc", "monster", "faction", "location", "object")}
    group = lambda r: r["key"]

tmp = Path(tempfile.mkdtemp(prefix="venn-e6-"))
S = DecisionService(model_dir=download_head(), cache_path=Path(a.cache).resolve(), registry_path=tmp / "registry.json")
t0 = time.time()
V = S.embed([r["state"] for r in items]); print(f"embedded {len(items)} states in {time.time()-t0:.0f}s", flush=True)
opts = [{"label": k, "vector": v.tolist()} for k, v in zip(LABELS, S.embed(list(LABELS.values())))]
iv = S.embed([instr])[0].tolist()
Q = {"type": "Choice", "instructions_vec": iv, "options": opts}

def probs(fam, idx):
    out = []
    for i in idx:
        ans = S.decide({"state": {"vector": V[i].tolist()}, "questions": {"k": {**Q, "family": fam}}})["answers"]["k"]
        out.append(ans["probabilities"])
    return out

groups = sorted({group(r) for r in items}); random.Random(1).shuffle(groups)
folds = [groups[i::a.folds] for i in range(a.folds)]
pooled, held, reports, preds = [], [], [], {}
for k, hg in enumerate(folds):
    hg = set(hg); te = [i for i, r in enumerate(items) if group(items[i]) in hg]; tr = [i for i in range(len(items)) if i not in set(te)]
    fam = f"e6-{a.task}-{k}"; t0 = time.time()
    # Venn rejects repeated states; the frozen sets repeat 7 + 4 rows verbatim (same label), so train on the first copy
    seen = set(); ex = []
    for i in tr:
        if items[i]["state"] in seen: continue
        seen.add(items[i]["state"]); ex.append({"state": V[i].tolist(), "label": label(items[i]), "group_id": group(items[i])})
    info = S.register_family(fam, ex, opts, iv, seed=0, form=a.form, class_weight="balanced" if a.balanced else None,
                             cv={"max_epochs": a.max_epochs} if a.max_epochs else None)
    reports.append({kk: info.get(kk) for kk in ("chosen", "settings", "cv_accuracy", "cv_ece", "temperature", "split_counts") if kk in info})
    pte = probs(fam, te)
    if a.task == "triage":
        ptr = [p["name"] for p, i in zip(probs(fam, tr), tr)]
        th = 0.5 * min(p for p, i in zip(ptr, tr) if items[i]["label"] == "name")   # zero misses on train, halved
        miss = [items[i]["token"] for p, i in zip(pte, te) if items[i]["label"] == "name" and p["name"] < th]
        held.append(dict(n=len(te), sent=sum(p["name"] >= th for p in pte), misses=miss))
        pooled += [(p["name"], items[i]["label"] == "name") for p, i in zip(pte, te)]
    else:
        for p, i in zip(pte, te): preds[i] = max(p, key=p.get)
    print(f"fold {k}: {len(tr)} train / {len(te)} held out, fit+score {time.time()-t0:.0f}s, report {reports[-1]}", flush=True)

print(f"== venn-1 family, form={a.form}, class_weight={'balanced' if a.balanced else 'none'}, max_epochs={a.max_epochs or 24}")
if a.task == "triage":
    pos = [p for p, l in pooled if l]; neg = [p for p, l in pooled if not l]
    auc = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg)); srt = sorted(pos)
    at = lambda k: sum(p >= srt[k] for p, _ in pooled) / len(pooled)
    print(f"tuned     pooled held-out AUC {auc:.3f} | forwards at 0 misses {at(0):.0%}, at <=2 misses {at(2):.0%}")
    n = sum(h["n"] for h in held); s = sum(h["sent"] for h in held); m = [x for h in held for x in h["misses"]]
    print(f"tuned     held-out: forwards {s}/{n} ({s/n:.0%}), misses {len(m)}: {m}")
else:
    c = sum(preds[i] == items[i]["gold"] for i in preds)
    agg = {}
    for st in ("repair", "control"):
        by = collections.defaultdict(set)
        for i, x in preds.items():
            if items[i]["set"] == st: by[items[i]["key"]].add(x)
        agg[st] = dict(entities=len(by), split=sum(len(v) > 1 for v in by.values()))
    ents = collections.defaultdict(list)
    for i, x in preds.items(): ents[(items[i]["key"], items[i]["gold"])].append(x)
    maj = sum(collections.Counter(xs).most_common(1)[0][0] == g for (key, g), xs in ents.items())
    print(f"tuned     held-out accuracy {c}/{len(preds)} ({c/len(preds):.1%}) | repair {agg['repair']} | control {agg['control']} | entity-majority correct {maj}/{len(ents)}")
    print("confusion (gold -> pred):", dict(collections.Counter((items[i]["gold"], x) for i, x in preds.items() if x != items[i]["gold"]).most_common(8)))
