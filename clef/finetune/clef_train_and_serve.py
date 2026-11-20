"""Train Clef's final readout on a frozen split, validate, then serve both heads.

Run inside clef-server with /trial mounted writable and HF cache read-only.
Stock `clef` remains available; `clef-card-tuned` is an experimental approval head.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

import torch
from safetensors.torch import load_file,save_file
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse
import uvicorn

from clef_readout import select_and_fit,evaluate

ROOT=Path(os.environ.get('TRIAL_ROOT','/trial'))
SNAPSHOT=Path('/root/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c')
QUESTION={'accept':{'type':'noul','instructions':"A tool proposed this correction to a D&D session transcript. Will the dungeon master approve it exactly as written, with no change and no need to discuss it?"}}
CODE_HASH=hashlib.sha256((SNAPSHOT/'joint_schema_model.py').read_bytes()).hexdigest()
sys.path.insert(0,str(SNAPSHOT))
import joint_schema_model as release


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def forward(model,processor,state):
    request={'model':'clef','state':state,'questions':QUESTION}
    encoded=release.encode_record(processor.tokenizer,request,max_length=16384,processor=processor)
    if len(encoded.input_ids)>=16384:raise ValueError('Input reached the token limit; do not train on silently shortened cards')
    assert encoded.questions[0].option_ids==('true','false')
    batch=release.collate_records([encoded],processor.tokenizer.pad_token_id,torch.device('cuda'))
    with torch.inference_mode():logits=model(batch)[0][0]
    return logits


def cache_split(model,processor,name):
    path=ROOT/f'{name}.json';metadata={'source_sha256':sha(path),'release_code_sha256':CODE_HASH,'question':QUESTION}
    destination=ROOT/f'{name}-features.pt'
    if destination.exists():
        cached=torch.load(destination,weights_only=False,map_location='cpu')
        assert cached['metadata']==metadata
        print('Reusing',name,'cached features',flush=True);return cached
    rows=read(path);vectors=[];bases=[];captured=[]
    def capture(module,inputs):captured.append(inputs[0].detach().float().cpu().clone())
    hook=model.head.residual_scorer[-1].register_forward_pre_hook(capture)
    gate=torch.sigmoid(model.head.residual_gate).float().item()
    try:
        for index,row in enumerate(rows):
            captured.clear();logits=forward(model,processor,row['features']['card_text'])
            assert len(captured)==1 and captured[0].shape==(2,1024)
            vectors.append(gate*(captured[0][0]-captured[0][1]))
            bases.append((logits[0].float()-logits[1].float()).item())
            if (index+1)%50==0 or index+1==len(rows):print(f'{name} features: {index+1}/{len(rows)}',flush=True)
    finally:hook.remove()
    cache={'metadata':metadata,'rows':rows,'x':torch.stack(vectors),'base_logit':torch.tensor(bases)}
    torch.save(cache,destination)
    return cache


print('Loading official Clef 27B; backbone remains frozen',flush=True)
model,processor=release.load_release_model(SNAPSHOT,device='cuda')
model.requires_grad_(False);model.eval()
stock_head=copy.deepcopy(model.head).eval()
checkpoint=ROOT/'trained_joint_head.safetensors'
report_path=ROOT/'report.json'
if report_path.exists() and checkpoint.exists():
    report=read(report_path)
    assert sha(checkpoint)==report['checkpoint_sha256']
    assert report['release_code_sha256']==CODE_HASH
    assert all(sha(ROOT/f'{s}.json')==report['data_hashes'][s] for s in ('development','holdout'))
    model.head.load_state_dict(load_file(checkpoint),strict=True)
else:
    development=cache_split(model,processor,'development')
    # Selection and final fitting happen before holdout labels are loaded.
    delta,selection=select_and_fit(development)
    torch.save(delta,ROOT/'trained_readout_delta.pt');write(ROOT/'selection-before-holdout.json',selection)
    assert delta.numel()==1024
    original=model.head.residual_scorer[-1].weight.detach().float().cpu()
    with torch.no_grad():model.head.residual_scorer[-1].weight.copy_((original+delta[None]).to(device='cuda',dtype=torch.bfloat16))
    save_file({k:v.detach().cpu().contiguous() for k,v in model.head.state_dict().items()},str(checkpoint))
    tuned_head=copy.deepcopy(model.head).eval()
    # Evaluate actual full-model forward passes, including BF16 rounding, for
    # both heads on the untouched holdout. Cached approximations are not reported
    # as heldout inference results.
    holdout=read(ROOT/'holdout.json')
    assert not ({r['session'] for r in development['rows']}&{r['session'] for r in holdout})
    predictions=[]
    for index,row in enumerate(holdout):
        scores={}
        for name,head in [('stock',stock_head),('tuned',tuned_head)]:
            model.head=head
            logits=forward(model,processor,row['features']['card_text'])
            scores[name]=logits.float().softmax(-1)[0].item()
        predictions.append({'record_id':row['record_id'],'session':row['session'],'id':row['id'],
                            'target':row['target'],'concrete_proposal':row['concrete_proposal'],**scores})
        if (index+1)%25==0 or index+1==len(holdout):print(f'Paired actual holdout inference: {index+1}/{len(holdout)}',flush=True)
    write(ROOT/'holdout-predictions.json',predictions)
    report={'method':'L2-anchored fine-tuning of the 1024 final residual-scorer weights; all other weights frozen',
            'release_snapshot':SNAPSHOT.name,'release_code_sha256':CODE_HASH,'selection':selection,
            'development_rows':len(development['rows']),'holdout_rows':len(holdout),
            'data_hashes':{s:sha(ROOT/f'{s}.json') for s in ('development','holdout')},
            'checkpoint_sha256':sha(checkpoint),'heldout':{name:evaluate(holdout,[r[name] for r in predictions]) for name in ('stock','tuned')},
            'notes':['Holdout labels were not used for training or hyperparameter selection.',
                     'Holdout scores are actual paired full-model forward passes.',
                     'Only four concrete non-approvals occur in the heldout set; treat comparisons cautiously.',
                     'The experimental head is task-specific; stock clef remains the default model ID.']}
    write(report_path,report);print(json.dumps(report,indent=2),flush=True)

tuned_head=copy.deepcopy(model.head).eval() if model.head is not stock_head else tuned_head
model.head=stock_head
heads={'clef':stock_head,'clef-card-tuned':tuned_head}
lock=threading.Lock()
app=FastAPI(docs_url=None,redoc_url=None)


@app.get('/health')
def health():return {'ready':True,'loaded':list(heads)}


@app.get('/v1/models')
def models():return {'object':'list','data':[{'id':name,'object':'model'} for name in heads]}


@app.post('/v1/systemone')
def answer(body:dict):
    name=body.get('model')
    if name not in heads:raise HTTPException(404,'Use clef or clef-card-tuned')
    if name=='clef-card-tuned' and (len(body.get('questions',{}))!=1 or any(q.get('type')!='noul' for q in body['questions'].values())):
        raise HTTPException(400,'The tuned head supports a single approval question')
    with lock:
        model.head=heads[name];started=time.monotonic()
        try:result=release.systemone(model,processor,body,max_length=16384)
        finally:model.head=stock_head
    return JSONResponse(result,headers={'X-Clef-Latency-Ms':f'{(time.monotonic()-started)*1000:.1f}'})


print('Training complete; serving stock clef and experimental clef-card-tuned on :8002',flush=True)
uvicorn.run(app,host='0.0.0.0',port=8002,log_level='warning')
