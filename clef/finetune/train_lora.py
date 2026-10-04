"""Stage 2: LoRA on Nox-4B's backbone + its head, valid-k CE, evaluated by held-out group.

Loads the model through vllm_sr_runtime exactly as `serve` does, wraps the MLP and attention
projections in the runtime's own LoRALinear (rank R), and trains LoRA + head with gradients
flowing through the native backbone (the runtime only ever runs it under inference_mode).
    python3 train_lora.py items.json --task triage|typing [--folds 3] [--rank 16] [--epochs 2]
"""
import argparse, collections, copy, json, math, random, time, torch
from torch import nn
from vllm_sr_runtime.config import ServeConfig
from vllm_sr_runtime.runtime import Runtime
from vllm_sr_runtime.families.decision2.renderer import collate
from vllm_sr_runtime.engines.native.models.lora import LoRALinear

ap = argparse.ArgumentParser()
ap.add_argument("items"); ap.add_argument("--task", required=True)
ap.add_argument("--folds", type=int, default=3); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--head-lr", type=float, default=1e-4); ap.add_argument("--pos-weight", type=float, default=4.0)
ap.add_argument("--max-train", type=int, default=0, help="subsample training items per fold (0 = all)")
a = ap.parse_args()
torch.manual_seed(0); random.seed(0)
TARGETS = ("gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj")

rt = Runtime(ServeConfig(model="vllm-sr/Decision-2.0-Nox-4B", device="cuda", offline=True)); rt.load()
m = rt.model; bb = m.engine_model.backbone; dev = m.engine_model.device; T = m.details.temperatures["choice"]
rt.scheduler.stop() if hasattr(rt.scheduler, "stop") else None
for p in bb.parameters(): p.requires_grad_(False)
base_head = copy.deepcopy(m.head).eval()

items = json.load(open(a.items))
def render(it):
    plan = m.plan(it["state"], it["questions"]); ri = plan.items[0]
    b = collate([ri], m.tokenizer.pad_id)
    return dict(b=b, keys=ri.keys, valid=[ri.keys.index(v) for v in it["valid"]], meta={k: it[k] for k in it if k not in ("state", "questions")})
R = [render(it) for it in items]
group = (lambda r: r["meta"]["session"]) if a.task == "triage" else (lambda r: r["meta"]["key"])

def install(rank):
    """Wrap target Linears in fresh LoRA; returns the LoRA modules (B=0, so the model starts unchanged)."""
    mods = []
    for name, mod in list(bb.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in TARGETS:
                lora = LoRALinear(child, rank, scaling=2.0).to(dev)
                nn.init.kaiming_uniform_(lora.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(lora.lora_B.weight)
                setattr(mod, cname, lora); mods.append(lora)
    return mods

def uninstall():
    for name, mod in list(bb.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, LoRALinear): setattr(mod, cname, child.base_layer)

def logits_of(r, head):
    b = r["b"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = bb(b["input_ids"].to(dev), b["attention_mask"].to(dev))
    k = len(r["keys"])
    g = hidden[0, b["candidate_positions"][0, :k].to(dev)]; q = hidden[0, b["query_positions"][0].to(dev)]
    return head(g[None], q[None])[0] / T

def train_fold(tr):
    mods = install(a.rank); head = copy.deepcopy(base_head).train()
    params = [p for l in mods for p in (l.lora_A.weight, l.lora_B.weight)]
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.0)
    if a.max_train: tr = random.sample(tr, min(a.max_train, len(tr)))
    t0 = time.time(); n = 0
    for ep in range(a.epochs):
        random.shuffle(tr)
        for r in tr:
            p = torch.softmax(logits_of(r, head), -1)
            w = a.pos_weight if (a.task == "triage" and r["meta"]["label"] == "name") else 1.0
            loss = -w * torch.log(p[r["valid"]].sum().clamp_min(1e-9))
            loss = loss + 1e-1 * sum(((p1 - p0) ** 2).sum() for p1, p0 in zip(head.parameters(), base_head.parameters()))
            opt.zero_grad(); loss.backward(); opt.step(); n += 1
            if n % 200 == 0: print(f"  step {n} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    return mods, head.eval()

@torch.no_grad()
def predict(rs, head):
    return [torch.softmax(logits_of(r, head), -1).float().cpu() for r in rs]

groups = sorted({group(r) for r in R}); random.Random(1).shuffle(groups)
folds = [groups[i::a.folds] for i in range(a.folds)]
out = collections.defaultdict(list)
for k, held in enumerate(folds):
    te = [r for r in R if group(r) in held]; tr = [r for r in R if group(r) not in held]
    print(f"fold {k}: train {len(tr)} test {len(te)}", flush=True)
    rel = predict(te, base_head)
    mods, head = train_fold(list(tr)); tun = predict(te, head); trp = predict(tr, head); uninstall()
    for name, P in (("released", rel), ("tuned", tun)):
        for p, r in zip(P, te): out[name].append((p, r))
    out["tuned_train"].append((trp, tr))
    torch.cuda.empty_cache()
torch.save(dict((k, v) for k, v in out.items() if k != "tuned_train"), f"lora_{a.task}_preds.pt")

if a.task == "triage":
    for name in ("released", "tuned"):
        P = [(p[r["keys"].index("campaign")].item(), r["meta"]["label"] == "name", r) for p, r in out[name]]
        pos = sorted(x for x, l, _ in P if l); neg = [x for x, l, _ in P if not l]
        auc = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg))
        at = lambda j: sum(x >= pos[j] for x, _, _ in P) / len(P)
        print(f"{name:9s} pooled held-out AUC {auc:.3f} | forwards at 0 misses {at(0):.0%}, at <=2 misses {at(2):.0%} | lowest names {[r['meta']['token'] for x,l,r in sorted(P,key=lambda t:t[0]) if l][:4]}")
else:
    ENT = {"npc", "monster", "faction", "location", "object"}
    for name in ("released", "tuned"):
        pr = [(r["keys"][p.argmax().item()], r) for p, r in out[name]]
        acc = sum(r["keys"].index(x) in r["valid"] for x, r in pr) / len(pr)
        msg = f"{name:9s} held-out accuracy {acc:.1%}"
        for st in ("repair", "control"):
            by = collections.defaultdict(set); mv = 0
            for x, r in pr:
                if r["meta"]["set"] != st: continue
                (by[r["meta"]["key"]].add(x) if x in ENT else None); mv += x not in ENT
            msg += f" | {st}: split {sum(len(v)>1 for v in by.values())}/{len(by)} moved {mv}"
        print(msg)
