"""Head-only fine-tune of Nox-4B, cross-fitted to the cascade replay's 2-fold split.

    python3 cascade_head.py features.pt plan.json out.json --task triage|typing

plan.json (from ../eval/cascade_ft.py plan): per feature row, `half` (0/1 = the cascade replay's
outer fold, None = training-only row) and `g` (fold group), plus `strict` (list of keys: in strict
mode a training row whose entity key also occurs in the test half is dropped).

For each outer half h, with the other half as training data:
  * test:  probabilities on half h from a head trained on all training rows,
  * oof:   probabilities on the training rows from an inner 3-fold (by group) cross-fit, so the
           replay can fit the parity gate on predictions that, like the test ones, never saw their
           own labels.
Also writes the released head's probabilities on every row (sanity check vs the served ones).
Same loss / temperature / L2-SP as train_head.py; settings fixed in advance (e6.sh's), not tuned.
"""
import argparse, copy, json, random, torch
from vllm_sr_runtime.families.decision2.readout import load_head

ap = argparse.ArgumentParser()
ap.add_argument("features"); ap.add_argument("plan"); ap.add_argument("out")
ap.add_argument("--task", required=True); ap.add_argument("--head", required=True)
ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--l2sp", type=float, default=0.1); ap.add_argument("--pos-weight", type=float, default=4.0)
ap.add_argument("--inner", type=int, default=3); ap.add_argument("--strict", action="store_true")
ap.add_argument("--save-prefix", help="also save each half's test head as PREFIX-h{0,1}.safetensors (for serve_tuned.py)")
a = ap.parse_args()
torch.manual_seed(0); random.seed(0)
D = torch.load(a.features, weights_only=False); F = D["features"]; T = D["temperatures"]["choice"]
P = json.load(open(a.plan))["rows"]; assert len(P) == len(F)
H = F[0]["g"].shape[-1]; DEV = "cpu"
base = load_head(a.head, H, 256).to(DEV).eval()
assert len({tuple(f["keys"]) for f in F}) == 1
KEYS = F[0]["keys"]
G = torch.stack([f["g"] for f in F]); Q = torch.stack([f["q"] for f in F])
V = torch.zeros(len(F), G.shape[1], dtype=torch.bool)
for i, f in enumerate(F): V[i, f["valid"]] = True
W = torch.tensor([a.pos_weight if (a.task == "triage" and f["meta"]["label"] == "name") else 1.0 for f in F])

def probs(head, idx):
    with torch.no_grad(): return torch.softmax(head(G[idx], Q[idx]) / T, -1)

def train(idx):
    torch.manual_seed(0); rnd = random.Random(0)
    head = copy.deepcopy(base).train(); idx = list(idx)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)
    for _ in range(a.epochs):
        rnd.shuffle(idx)
        for s in range(0, len(idx), 32):
            b = torch.tensor(idx[s:s + 32]); opt.zero_grad()
            p = torch.softmax(head(G[b], Q[b]) / T, -1); lp = torch.log((p * V[b]).sum(-1).clamp_min(1e-9))
            reg = sum(((p1 - p0.detach()) ** 2).sum() for p1, p0 in zip(head.parameters(), base.parameters()))
            (-(W[b] * lp).mean() + a.l2sp * reg).backward(); opt.step()
    return head.eval()

def pdict(head, idx):
    return {i: dict(zip(KEYS, p.tolist())) for i, p in zip(idx, probs(head, torch.tensor(idx)))}

def inner_folds(idx):
    gs = sorted({P[i]["g"] for i in idx}); random.Random(1).shuffle(gs)
    return [{g for g in gs[k::a.inner]} for k in range(a.inner)]

out = dict(task=a.task, strict=a.strict, settings=vars(a), keys=KEYS,
           released=pdict(base, list(range(len(F)))), halves={})
for h in (0, 1):
    test = [i for i, r in enumerate(P) if r["half"] == h]
    tkeys = {P[i]["key"] for i in test}
    tr = [i for i, r in enumerate(P) if r["half"] != h and not (a.strict and r["key"] in tkeys)]
    oof = {}
    for held in inner_folds([i for i in tr if P[i]["half"] is not None]):
        hi = [i for i in tr if P[i]["half"] is not None and P[i]["g"] in held]
        hk = {P[i]["key"] for i in hi}
        ti = [i for i in tr if P[i]["g"] not in held and not (a.strict and P[i]["key"] in hk)]
        oof.update(pdict(train(ti), hi))
    final = train(tr)
    out["halves"][h] = dict(n_train=len(tr), test=pdict(final, test), oof=oof)
    if a.save_prefix:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in final.state_dict().items()}, f"{a.save_prefix}-h{h}.safetensors")
    print(f"half {h}: train {len(tr)}, test {len(test)}, oof {len(oof)}", flush=True)
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
