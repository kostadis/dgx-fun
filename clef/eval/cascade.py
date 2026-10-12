"""Offline "System One Auto" cascade: a small decision model answers, low-confidence items escalate.

    python3 cascade.py [--small nox] [--big qwen] [--parity DELTA] [--json OUT] [--gates OUT]
    DATA=~/data/decision-eval/v1 (default)

vllm-sr's System One Auto (vllm-sr.ai/blog/system-one-auto) runs a small decision model first and
re-asks the larger model when the small model's top probability is below a gate threshold. Both
models here were already run on every item of the frozen set (temperature 0 / deterministic heads),
so the cascade is an exact function of the saved per-item answers: no new inference is needed.
cascade_live.py runs the same cascade against the real endpoints to test that claim.

The gate threshold is chosen 2-fold by session (fit on half the sessions, applied to the other half,
the split score.py's E3 held-out row uses), so the reported accuracy never sees its own labels.
The "oracle" row picks the threshold on the test items themselves -- an upper bound only.

Latency: cascade ms = small ms + big ms on escalated items. The frozen qwen runs were 8-way
concurrent, so its per-request ms is inflated relative to single-stream; cascade_paper.py redoes the
latency arithmetic with live single-stream service times.
"""
import argparse, json, os, random, statistics
from pathlib import Path

DATA = Path(os.path.expanduser(os.environ.get('DATA', '~/data/decision-eval/v1')))
load = lambda part, m: json.load(open(DATA / 'results' / part / f'{m}.json'))
top = lambda d: max(d, key=d.get)


def e2(r):
    return r['gold'] if r['gold_in_opts'] else 'OTHER', r['ans'].get('choice'), max(r['ans']['probabilities'].values())
def e2b(r): return r['ans'].get('choice')

def e3(r):  # binary: is this token a campaign name? NOT the paper's E3 metric: 628/864 are 'leave', so
    # plain accuracy rewards a gate that keeps confident "not a name" answers and drops real names
    # (kai->qwen "beats" qwen here only by missing ~half the 84 names qwen catches). Read with care.
    return r['label'] == 'name', top(r['p']) == 'campaign', max(r['p']['campaign'], 1 - r['p']['campaign'])
def e3b(r): return top(r['p']) == 'campaign'

def e4(r): return r['gold'], top(r['p']), max(r['p'].values())
def e4b(r): return top(r['p'])

TASKS = {  # part -> (session key, small extractor, big extractor, row filter)
    'e2':       ('session', e2, e2b, lambda r: True),
    'e3':       ('session', e3, e3b, lambda r: True),
    'e4_ent':   ('key',     e4, e4b, lambda r: r['status'] == 'merged'),   # score.py's E4 entity set
    'e4_facts': ('chapter', e4, e4b, lambda r: True),
}


def rows(part, small, big):
    """One dict per scored row: i = row index in the frozen file, g = fold group, gold, small
    prediction/confidence/ms, big prediction/ms."""
    sk, fs, fb, keep = TASKS[part]
    S, B = load(part, small), load(part, big)
    assert len(S) == len(B)
    out = []
    for i, (s, b) in enumerate(zip(S, B)):
        assert all(s.get(k) == b.get(k) for k in ('session', 'id', 'token', 'key', 'chapter')), 'row mismatch'
        if 'error' in s or 'error' in b or not keep(s): continue
        gold, sp, sc = fs(s)
        out.append(dict(i=i, g=s[sk], gold=gold, sp=sp, sc=sc, sms=s['ms'], bp=fb(b), bms=b['ms']))
    return out


def run(rs, t):
    esc = [r['sc'] < t for r in rs]
    pred = [r['bp'] if e else r['sp'] for r, e in zip(rs, esc)]
    ok = [p == r['gold'] for p, r in zip(pred, rs)]
    ms = [r['sms'] + (r['bms'] if e else 0) for r, e in zip(rs, esc)]
    return ok, esc, ms


def missed(rs, ok):  # e3 only: real names the answer drops (gold True = name)
    return sum(1 for r, o in zip(rs, ok) if r['gold'] is True and not o)


def best_t(rs, parity=None):
    """Threshold maximising accuracy; ties -> fewest escalations (lowest t). With parity=DELTA: the
    lowest threshold that stays within DELTA points of the big model alone (and on e3 misses no more
    names than it) -- escalating everything always qualifies."""
    cands = sorted({0.0, 1.01} | {r['sc'] + 1e-9 for r in rs})
    if parity is None:
        return max(cands, key=lambda t: (sum(run(rs, t)[0]), -t))
    ok_b = [r['bp'] == r['gold'] for r in rs]
    for t in cands:
        ok = run(rs, t)[0]
        if 100 * (sum(ok) - sum(ok_b)) / len(rs) >= -parity and missed(rs, ok) <= missed(rs, ok_b):
            return t
    return 1.01


def folds(rs):
    groups = sorted({r['g'] for r in rs}); halves = [set(groups[0::2]), set(groups[1::2])]
    return ((halves[0], halves[1]), (halves[1], halves[0]))


def heldout(rs, parity=None):
    """2-fold held-out cascade. Returns per-row (in rs order) ok, esc, ms, and the fold threshold."""
    n = len(rs); ok, esc, ms, th = [None] * n, [None] * n, [None] * n, [None] * n
    for tr, te in folds(rs):
        t = best_t([r for r in rs if r['g'] in tr], parity)
        idx = [i for i, r in enumerate(rs) if r['g'] in te]
        o, e, m = run([rs[i] for i in idx], t)
        for j, i in enumerate(idx): ok[i], esc[i], ms[i], th[i] = o[j], e[j], m[j], t
    return ok, esc, ms, th


def ci(diff, n=2000, seed=0):
    rnd = random.Random(seed); k = len(diff)
    bs = sorted(sum(diff[rnd.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return bs[int(.025 * n)], bs[int(.975 * n)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--small', default='nox'); ap.add_argument('--big', default='qwen')
    ap.add_argument('--json')
    ap.add_argument('--gates', metavar='OUT', help='write the held-out gate (per row index of the frozen file: fold '
                    'threshold, predicted escalation and prediction) so a live cascade can apply the same gates and be '
                    'compared row by row')
    ap.add_argument('--parity', type=float, metavar='DELTA',
                    help='pick the lowest threshold (fewest escalations) whose train-fold accuracy is within DELTA points of '
                         'the big model alone (and, on e3, that misses no more names than it), instead of maximising accuracy')
    a = ap.parse_args()
    OUT, GATES = {}, {}
    for part in TASKS:
        rs = rows(part, a.small, a.big); n = len(rs)
        if not rs: continue
        ok_s = [r['sp'] == r['gold'] for r in rs]; ok_b = [r['bp'] == r['gold'] for r in rs]
        ok_c, esc_c, ms_c, th = heldout(rs, a.parity)
        for r, e, t in zip(rs, esc_c, th):
            GATES.setdefault(part, {})[r['i']] = dict(t=t, esc=e, pred=r['bp'] if e else r['sp'])
        ts = [th[next(i for i, r in enumerate(rs) if r['g'] in te)] for _, te in folds(rs)]
        oracle = sum(run(rs, best_t(rs))[0])
        vs_s = ci([int(c) - int(s) for c, s in zip(ok_c, ok_s)]); vs_b = ci([int(c) - int(b) for c, b in zip(ok_c, ok_b)])
        fixed = sum(e and c and not s for e, c, s in zip(esc_c, ok_c, ok_s))
        broke = sum(e and s and not c for e, c, s in zip(esc_c, ok_c, ok_s))
        med = lambda xs: round(statistics.median(xs))
        res = dict(n=n, small=f'{sum(ok_s) / n:.1%}', big=f'{sum(ok_b) / n:.1%}', cascade=f'{sum(ok_c) / n:.1%}',
                   escalated=f'{sum(esc_c) / n:.0%}', thresholds=[round(t, 3) for t in ts],
                   fixed=fixed, broke=broke,
                   d_vs_small=f'{100 * (sum(ok_c) - sum(ok_s)) / n:+.1f}pt [{100 * vs_s[0]:+.1f},{100 * vs_s[1]:+.1f}]',
                   d_vs_big=f'{100 * (sum(ok_c) - sum(ok_b)) / n:+.1f}pt [{100 * vs_b[0]:+.1f},{100 * vs_b[1]:+.1f}]',
                   oracle=f'{oracle / n:.1%}',
                   missed_names=dict(small=missed(rs, ok_s), big=missed(rs, ok_b), cascade=missed(rs, ok_c)) if part == 'e3' else None,
                   both_wrong=sum(not s and not b for s, b in zip(ok_s, ok_b)),
                   med_ms=dict(small=med([r['sms'] for r in rs]), big=med([r['bms'] for r in rs]), cascade=med(ms_c)),
                   mean_ms=dict(small=round(statistics.mean(r['sms'] for r in rs)), big=round(statistics.mean(r['bms'] for r in rs)),
                                cascade=round(statistics.mean(ms_c))))
        OUT[part] = res
        print(f'== {part}  ({a.small} -> {a.big}, n={n})')
        for k, v in res.items(): print(f'  {k:11s} {v}')
        print('  sweep (all items, NOT held out):  t  esc  acc')
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            o, e, _ = run(rs, t); print(f'    {t:.2f}  {sum(e) / n:4.0%}  {sum(o) / n:.1%}')
    if a.json: json.dump(OUT, open(a.json, 'w'), indent=1)
    if a.gates: json.dump(dict(small=a.small, big=a.big, parity=a.parity, gates=GATES), open(a.gates, 'w'), indent=1)


if __name__ == '__main__':
    main()
