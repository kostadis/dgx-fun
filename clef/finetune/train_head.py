"""Stage 1: train only Nox's candidate head on cached features; evaluate by held-out group.

Loss = valid-k cross-entropy (-log sum of probabilities of the acceptable options), with
the server's own temperature, positive-class weighting, and an L2 pull toward the released
head (L2-SP) so the head cannot drift far from the original.
    python3 train_head.py features.pt --task triage|typing [--folds 3] [--save head.safetensors]
"""
import argparse, copy, json, math, random, collections, torch
from vllm_sr_runtime.families.decision2.readout import CandidateHead, load_head

ap = argparse.ArgumentParser()
ap.add_argument("features"); ap.add_argument("--task", required=True)
ap.add_argument("--head", required=True, help="released decision_head.safetensors")
ap.add_argument("--folds", type=int, default=3); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--l2sp", type=float, default=1e-2)
ap.add_argument("--pos-weight", type=float, default=4.0); ap.add_argument("--save")
a = ap.parse_args()
torch.manual_seed(0); random.seed(0)
D = torch.load(a.features, weights_only=False); F = D["features"]; T = D["temperatures"]["choice"]
H = F[0]["g"].shape[-1]; DEV = "cuda" if torch.cuda.is_available() else "cpu"
base = load_head(a.head, H, 256).to(DEV)
assert len({tuple(f["keys"]) for f in F}) == 1, "batched trainer needs one shared option list"
G = torch.stack([f["g"] for f in F]).to(DEV); Qf = torch.stack([f["q"] for f in F]).to(DEV)
V = torch.zeros(len(F), G.shape[1], dtype=torch.bool, device=DEV)
for i, f in enumerate(F):
    f["i"] = i; V[i, f["valid"]] = True
W = torch.tensor([a.pos_weight if (a.task == "triage" and f["meta"]["label"] == "name") else 1.0 for f in F], device=DEV)
group = (lambda f: f["meta"]["session"]) if a.task == "triage" else (lambda f: f["meta"]["key"])

def bprobs(head, idx):
    return torch.softmax(head(G[idx], Qf[idx]) / T, -1)

def loss_fn(head, batch):
    idx = torch.tensor([f["i"] for f in batch], device=DEV)
    p = bprobs(head, idx); lp = torch.log((p * V[idx]).sum(-1).clamp_min(1e-9))
    reg = sum(((p1 - p0.detach()) ** 2).sum() for p1, p0 in zip(head.parameters(), base.parameters()))
    return -(W[idx] * lp).mean() + a.l2sp * reg

def train(fs):
    head = copy.deepcopy(base).train()
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)
    for ep in range(a.epochs):
        random.shuffle(fs)
        for i in range(0, len(fs), 32):
            opt.zero_grad(); l = loss_fn(head, fs[i:i + 32]); l.backward(); opt.step()
    return head.eval()

@torch.no_grad()
def score(head, fs):
    return list(bprobs(head, torch.tensor([f["i"] for f in fs], device=DEV)).cpu())

def triage_eval(head, tr, te):
    ci = lambda f: f["keys"].index("campaign")
    ptr = [p[ci(f)].item() for p, f in zip(score(head, tr), tr)]
    names = [p for p, f in zip(ptr, tr) if f["meta"]["label"] == "name"]
    th = 0.5 * min(names)                      # zero misses on train, halved for margin
    pte = [p[ci(f)].item() for p, f in zip(score(head, te), te)]
    sent = sum(p >= th for p in pte); miss = [f["meta"]["token"] for p, f in zip(pte, te) if f["meta"]["label"] == "name" and p < th]
    return dict(n=len(te), sent=sent, misses=miss, threshold=th)

ENTITY_DUMP = {}
ENT = {"npc", "monster", "faction", "location", "object"}
def typing_eval(head, tr, te):
    preds = [f["keys"][p.argmax().item()] for p, f in zip(score(head, te), te)]
    out = dict(n=len(te), correct=sum(f["keys"].index(x) in f["valid"] for x, f in zip(preds, te)))
    ENTITY_DUMP.setdefault(head is base, {}).update({f["meta"]["key"]: ENTITY_DUMP[head is base].get(f["meta"]["key"], []) + [x] for x, f in zip(preds, te)} if False else {})
    for x, f in zip(preds, te): ENTITY_DUMP.setdefault("released" if head is base else "tuned", collections.defaultdict(list))[(f["meta"]["key"], f["meta"]["gold"])].append(x)
    for st in ("repair", "control"):
        by = collections.defaultdict(set); moved = 0
        for x, f in zip(preds, te):
            if f["meta"]["set"] != st: continue
            if x in ENT: by[f["meta"]["key"]].add(x)
            else: moved += 1
        out[st] = dict(entities=len(by), split=sum(len(v) > 1 for v in by.values()), moved=moved)
    return out

ev = triage_eval if a.task == "triage" else typing_eval
groups = sorted({group(f) for f in F}); random.Random(1).shuffle(groups)
folds = [groups[i::a.folds] for i in range(a.folds)]
res = collections.defaultdict(list); pooled = collections.defaultdict(list)
for k, held in enumerate(folds):
    te = [f for f in F if group(f) in held]; tr = [f for f in F if group(f) not in held]
    tuned = train(list(tr))
    res["released"].append(ev(base.eval(), tr, te)); res["tuned"].append(ev(tuned, tr, te))
    if a.task == "triage":
        for nm, hd in (("released", base.eval()), ("tuned", tuned)):
            for p, f in zip(score(hd, te), te):
                pooled[nm].append((p[f["keys"].index("campaign")].item(), f["meta"]["label"] == "name"))
    print(f"fold {k}: released {res['released'][-1]} | tuned {res['tuned'][-1]}", flush=True)
if a.task == "triage":
    for name in pooled:
        P = pooled[name]; pos = [p for p, l in P if l]; neg = [p for p, l in P if not l]
        auc = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))
        srt = sorted(pos)
        at = lambda k: sum(p >= srt[k] for p, _ in P) / len(P)
        print(f"{name:9s} pooled held-out AUC {auc:.3f} | forwards at 0 misses {at(0):.0%}, at <=2 misses {at(2):.0%}")
for name, rs in res.items():
    if a.task == "triage":
        n = sum(r["n"] for r in rs); s = sum(r["sent"] for r in rs); m = sum(len(r["misses"]) for r in rs)
        print(f"{name:9s} held-out: forwards {s}/{n} ({s/n:.0%}), misses {m}: {[x for r in rs for x in r['misses']]}")
    else:
        n = sum(r["n"] for r in rs); c = sum(r["correct"] for r in rs)
        agg = {st: {k: sum(r[st][k] for r in rs) for k in ("entities", "split", "moved")} for st in ("repair", "control")}
        print(f"{name:9s} held-out accuracy {c}/{n} ({c/n:.1%}) | repair {agg['repair']} | control {agg['control']}")
if a.task == "typing":
    rows = []
    for (key, gold), xs in ENTITY_DUMP.get("tuned", {}).items():
        rel = ENTITY_DUMP["released"][(key, gold)]
        rows.append(dict(key=key, gold=gold, n=len(xs), tuned=collections.Counter(xs).most_common(), released=collections.Counter(rel).most_common()))
    json.dump(rows, open("typing_entity_preds.json", "w"), indent=1)
if a.save:
    from safetensors.torch import save_file
    final = train(list(F)); save_file({k: v.contiguous() for k, v in final.state_dict().items()}, a.save); print("saved", a.save)
