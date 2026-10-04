"""Run Nox-4B's frozen backbone once over labelled items; cache option/query features.

Loads the model through vllm_sr_runtime exactly as `serve` does, renders each item
with the runtime's own planner, and saves the hidden rows the head reads.
    python3 extract.py items.json features.pt [model]
"""
import json, sys, torch
from vllm_sr_runtime.config import ServeConfig
from vllm_sr_runtime.runtime import Runtime
from vllm_sr_runtime.families.decision2.renderer import collate
from vllm_sr_runtime.plugins.base import ForwardBatch
from vllm_sr_runtime.families.decision2.readout import logits

items_path, out_path = sys.argv[1], sys.argv[2]
model_id = sys.argv[3] if len(sys.argv) > 3 else "vllm-sr/Decision-2.0-Nox-4B"
rt = Runtime(ServeConfig(model=model_id, device="cuda", offline=True)); rt.load()
m = rt.model
T = m.details.temperatures
items = json.load(open(items_path)); feats = []
for n, it in enumerate(items):
    plan = m.plan(it["state"], it["questions"])
    assert not plan.errors and len(plan.items) == 1, plan.errors
    ri = plan.items[0]
    b = collate([ri], m.tokenizer.pad_id)
    out = m.engine_model.forward(ForwardBatch(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
        gather=b["candidate_positions"], query=b["query_positions"], lengths=[len(ri.ids)], shared_prefix=0))
    k = len(ri.keys)
    g = out.gathered[0, :k].float().cpu().clone(); q = out.query[0].float().cpu().clone()
    with torch.no_grad():
        lg = m.head(g[None].to(m.head.key.weight.device), q[None].to(m.head.key.weight.device))[0].cpu()
    feats.append(dict(g=g, q=q, keys=ri.keys, valid=[ri.keys.index(v) for v in it["valid"]],
                      task=ri.task_type, logits0=lg, meta={k2: it[k2] for k2 in it if k2 not in ("state", "questions")}))
    if n % 100 == 0: print(n, flush=True)
torch.save(dict(features=feats, temperatures=T, model=model_id), out_path)
print("saved", len(feats), "items; temperatures", T)
rt.scheduler.stop() if hasattr(rt.scheduler, "stop") else None
