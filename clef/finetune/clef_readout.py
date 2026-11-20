"""Regularized final-readout training on cached Clef joint-head features.

Only 1024 final linear weights change. Backbone, routing, decoder, bias and
scales stay frozen. No holdout labels are used in cross-validation or fitting.
"""
import collections
import math
import random


def auc(labels, probabilities):
    pos=[p for p,y in zip(probabilities,labels) if y]
    neg=[p for p,y in zip(probabilities,labels) if not y]
    if not pos or not neg:return None
    return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))


def evaluate(rows, probabilities):
    assert len(rows)==len(probabilities)
    groups={k:{'cards':0,'nonapprovals':0} for k in ('over90','80to90','regular')}
    y=[];p=[]
    for row,score in zip(rows,probabilities):
        assert math.isfinite(score) and 0<=score<=1
        concrete=row['concrete_proposal'];label=row['target']['approve_as_presented']
        group='over90' if concrete and score>.9 else '80to90' if concrete and score>=.8 else 'regular'
        groups[group]['cards']+=1;groups[group]['nonapprovals']+=not label
        if concrete:y.append(label);p.append(score)
    nll=-sum(math.log(max(1e-9,s if label else 1-s)) for s,label in zip(p,y))/max(1,len(y))
    return {'rows':len(rows),'concrete_proposals':len(y),'concrete_nonapprovals':y.count(False),
            'concrete_auc':auc(y,p),'concrete_log_loss':nll,'groups':groups}


def folds(rows,count=3):
    sessions=sorted({r['session'] for r in rows});random.Random(7).shuffle(sessions)
    for k in range(count):
        held=set(sessions[k::count])
        yield ([i for i,r in enumerate(rows) if r['session'] not in held],
               [i for i,r in enumerate(rows) if r['session'] in held])


def fit(features,base,rows,indices,l2,max_iter=100):
    import torch
    import torch.nn.functional as F
    x=features[indices].double();b=base[indices].double()
    y=torch.tensor([float(rows[i]['target']['approve_as_presented']) for i in indices],dtype=torch.float64)
    # Open questions stay on regular cards regardless of score; retain them as
    # weak auxiliary examples rather than letting them dominate the actual task.
    weights=torch.tensor([1.0 if rows[i]['concrete_proposal'] else .25 for i in indices],dtype=torch.float64)
    delta=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([delta],lr=1,max_iter=max_iter,line_search_fn='strong_wolfe',tolerance_grad=1e-8)
    def closure():
        optimizer.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(b+x@delta,y,reduction='none')*weights).sum()/weights.sum()+.5*l2*delta.square().sum()
        loss.backward();return loss
    optimizer.step(closure)
    return delta.detach().float()


def select_and_fit(cache,lambdas=(.001,.01,.1,1.0,10.0)):
    import torch
    torch.set_num_threads(4)
    rows=cache['rows'];x=cache['x'];base=cache['base_logit']
    baseline=evaluate(rows,torch.sigmoid(base).tolist())
    candidates=[]
    for l2 in lambdas:
        predictions=[None]*len(rows)
        for train,validation in folds(rows):
            delta=fit(x,base,rows,train,l2)
            scores=torch.sigmoid(base[validation]+x[validation]@delta).tolist()
            for index,score in zip(validation,scores):predictions[index]=score
        metrics=evaluate(rows,predictions)
        candidates.append({'l2':l2,'out_of_fold':metrics})
        print(f'CV l2={l2}: AUC={metrics["concrete_auc"]:.4f} loss={metrics["concrete_log_loss"]:.4f}',flush=True)
    # Decide before opening any holdout labels. Prefer a stronger anchor when
    # discrimination ties; log loss breaks only AUC ties, never holdout results.
    best=max(candidates,key=lambda r:(r['out_of_fold']['concrete_auc'],-r['out_of_fold']['concrete_log_loss'],r['l2']))
    delta=fit(x,base,rows,list(range(len(rows))),best['l2'])
    return delta,{'trainable_parameters':x.shape[1],'cross_validation':'3 folds grouped by session; seed 7',
                  'development_baseline':baseline,'candidates':candidates,'selected_l2':best['l2'],
                  'selected_out_of_fold':best['out_of_fold'],
                  'development_improved_auc':best['out_of_fold']['concrete_auc']>baseline['concrete_auc']}
