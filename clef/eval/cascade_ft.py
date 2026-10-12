"""Replay a cascade whose first stage is a head-tuned Nox-4B (../finetune/cascade_head.py).

    python3 cascade_ft.py plan  DIR     # write DIR/plan-{triage,typing}.json for the container run
    python3 cascade_ft.py score DIR     # replay released vs tuned Nox -> qwen; DIR/ft-cascade.json
    python3 cascade_ft.py gates DIR     # freeze live gates per variant/half + DIR/prereg.json

Same replay as cascade_paper.py's SIM rows (parity gate, delta 0, qwen single-stream service times
from the live c=1 run), with one change: the small model's answer/confidence comes from a tuned head.
Nesting: for test half h the head was trained on the other half only, and the gate for h is fitted
on the other half's *inner out-of-fold* tuned predictions, so no item's label reaches its own head
or its own threshold. Nox's per-item latency is unchanged (same backbone, same head shape).
"""
import json, os, statistics, sys
from pathlib import Path
import cascade as C

DATA = C.DATA; LIVE = DATA / 'cascade' / 'live'
PARTS = {'e3': 'triage', 'e4_facts': 'typing'}


def qwen1(part):
    d = json.load(open(LIVE / part / 'big-c1.json'))
    return {r['i']: r['ms'] for r in d['rows'] if 'error' not in r}


def base_rows(part):
    q = qwen1(part)
    return [dict(r, bms=q[r['i']]) for r in C.rows(part, 'nox', 'qwen') if r['i'] in q]


def plan(out):
    for part, task in PARTS.items():
        rs = base_rows(part); half = {}
        for h, (_, te) in enumerate(C.folds(rs)):
            for r in rs:
                if r['g'] in te: half[r['i']] = h
        g = {r['i']: r['g'] for r in rs}
        items = json.load(open(DATA / f'e6_{task}_items.json'))
        P = [dict(half=half.get(i), g=g.get(i), key=(it['key'] if task == 'typing' else it['token'].lower()))
             for i, it in enumerate(items)]
        json.dump(dict(part=part, rows=P), open(Path(out) / f'plan-{task}.json', 'w'))
        print(part, task, len(P), 'rows,', sum(p['half'] is not None for p in P), 'in the cascade')


def small(part, p):  # (prediction, gate confidence) from a probability dict, as cascade.e3/e4 do
    if part == 'e3':
        top = max(p, key=p.get); return top == 'campaign', max(p['campaign'], 1 - p['campaign'])
    top = max(p, key=p.get); return top, p[top]


def summarize(part, rs, ok, esc, ms):
    n = len(rs); ok_b = [r['bp'] == r['gold'] for r in rs]; ok_s = [r['sp'] == r['gold'] for r in rs]
    lo, hi = C.ci([int(x) - int(y) for x, y in zip(ok, ok_b)])
    big_ms = statistics.mean(r['bms'] for r in rs); m = statistics.mean(ms)
    return dict(n=n, small_acc=round(100 * sum(ok_s) / n, 1), big_acc=round(100 * sum(ok_b) / n, 1),
                cascade_acc=round(100 * sum(ok) / n, 1), d_big=round(100 * (sum(ok) - sum(ok_b)) / n, 1),
                ci=[round(100 * lo, 1), round(100 * hi, 1)], esc=round(100 * sum(esc) / n, 1),
                mean_ms=round(m), big_ms=round(big_ms), speedup=round(big_ms / m, 2),
                missed=C.missed(rs, ok) if part == 'e3' else None,
                big_missed=C.missed(rs, ok_b) if part == 'e3' else None,
                small_missed=C.missed(rs, ok_s) if part == 'e3' else None)


def tuned_heldout(part, rs, F):
    """rs with the released Nox; F = cascade_head.py output. Returns summary + per-row records."""
    by = {r['i']: r for r in rs}; ok, esc, ms, test_rows, per = [], [], [], [], []
    for h in (0, 1):
        H = F['halves'][str(h)]
        tr = [dict(by[int(i)], **dict(zip(('sp', 'sc'), small(part, p)))) for i, p in H['oof'].items() if int(i) in by]
        te = [dict(by[int(i)], **dict(zip(('sp', 'sc'), small(part, p)))) for i, p in H['test'].items() if int(i) in by]
        t = C.best_t(tr, 0)
        o, e, m = C.run(te, t)
        ok += o; esc += e; ms += m; test_rows += te
        per += [dict(i=r['i'], half=h, t=t, sp=r['sp'], sc=r['sc'], esc=x) for r, x in zip(te, e)]
    return summarize(part, test_rows, ok, esc, ms), per


def score(d):
    d = Path(d); OUT = {}
    for part, task in PARTS.items():
        rs = base_rows(part)
        ok, esc, ms, _ = C.heldout(rs, 0)
        OUT[f'{part}/released'] = summarize(part, rs, ok, esc, ms)
        for mode in ('lenient', 'strict'):
            f = d / f'{task}-{mode}.json'
            if not f.exists(): continue
            F = json.load(open(f))
            # sanity: the released head on cached features reproduces the served probabilities
            S = C.load(part, 'nox')
            dp = [abs(F['released'][str(r['i'])][k] - S[r['i']]['p'][k]) for r in rs for k in S[r['i']]['p']]
            res, per = tuned_heldout(part, rs, F)
            res['released_feature_max_dp'] = round(max(dp), 4)
            OUT[f'{part}/tuned-{mode}'] = res
            json.dump(per, open(d / f'gates-{task}-{mode}.json', 'w'))
    for k, v in OUT.items():
        print(f'{k:22s}', ' '.join(f'{a}={b}' for a, b in v.items()))
    json.dump(OUT, open(d / 'ft-cascade.json', 'w'), indent=1)



def gates(d):
    """Freeze live gates (cascade_live.py format) per tuned variant and half, plus a pre-registration
    record: sha256 of every gates file and served head, and the replay's predictions, written before the
    first live request."""
    import hashlib, time
    d = Path(d); sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    pred = json.load(open(d / 'ft-cascade.json')); reg = dict(written=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runs={})
    LIVE_VARIANTS = {'e3': 'triage-lenient', 'e4_facts': 'typing-strict'}
    for part, var in LIVE_VARIANTS.items():
        per = json.load(open(d / f'gates-{var}.json'))
        allg = {}
        for h in (0, 1):
            g = {r['i']: dict(t=r['t'], esc=r['esc'], pred=r['sp']) for r in per if r['half'] == h}
            allg.update(g)
            f = d / f'live-gates-{part}-{var}-h{h}.json'
            json.dump(dict(small='nox', big='qwen', parity=0, variant=var, half=h, gates={part: g}), open(f, 'w'), indent=1)
            head = d / f'head-{var}-h{h}.safetensors'
            reg['runs'][f'{part}/{var}/h{h}'] = dict(gates=f.name, gates_sha256=sha(f), n=len(g),
                                                     head=head.name, head_sha256=sha(head) if head.exists() else None)
        f = d / f'live-gates-{part}-all.json'   # every row, for the qwen-alone runs
        json.dump(dict(small='nox', big='qwen', gates={part: allg}), open(f, 'w'), indent=1)
        reg['runs'][f'{part}/all'] = dict(gates=f.name, gates_sha256=sha(f), n=len(allg))
        reg[f'{part}/predicted'] = pred[f'{part}/tuned-{var.split("-")[1]}']
    json.dump(reg, open(d / 'prereg.json', 'w'), indent=1)
    print(json.dumps(reg, indent=1)); print('prereg sha256', sha(d / 'prereg.json'))


if __name__ == '__main__':
    {'plan': plan, 'score': score, 'gates': gates}[sys.argv[1]](sys.argv[2])
