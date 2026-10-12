"""Replay Vela 2.0 members as the cascade's first stage, beside released and tuned Nox (cascade_ft.py).

    python3 vela_replay.py [OUT.json]       # default $DATA/results/e9/vela-cascade.json

Same replay as cascade_ft.py's released rows: parity gate (delta 0, and on E3 no more missed names than qwen),
fitted 2-fold by group; first-stage answers, confidences and single-stream ms from the model's frozen run
(run.py), qwen's from the cascade study's live c=1 run. Each member is scored twice:
  plain  gate confidence as for Nox (top probability over the real options; max(p, 1-p) on E3),
  abs    the same times (1 - P[ABS]) -- the top option's share of the softmax that includes [ABS], so an
         item Vela would rather abstain on escalates. Only for members served with [ABS] in the question.
"""
import json, sys
from pathlib import Path
import cascade as C
import cascade_ft as F

MEMBERS = ['nox', 'vela-0.8b', 'vela-4b', 'vela-9b', 'vela-4b-noabs']


def rows(part, model):
    q = F.qwen1(part); S = C.load(part, model)
    rs = [dict(r, bms=q[r['i']]) for r in C.rows(part, model, 'qwen') if r['i'] in q]
    for r in rs:
        r['abs'] = S[r['i']].get('abs')
    return rs


def main(out):
    OUT = {}
    for part in F.PARTS:
        for m in MEMBERS:
            try:
                rs = rows(part, m)
            except FileNotFoundError:
                continue
            ok, esc, ms, _ = C.heldout(rs, 0)
            OUT[f'{part}/{m}'] = F.summarize(part, rs, ok, esc, ms)
            if all(r['abs'] is not None for r in rs):
                ra = [dict(r, sc=r['sc'] * (1 - r['abs'])) for r in rs]
                ok, esc, ms, _ = C.heldout(ra, 0)
                OUT[f'{part}/{m}+abs'] = F.summarize(part, ra, ok, esc, ms)
                OUT[f'{part}/{m}+abs']['abs_mean'] = round(sum(r['abs'] for r in rs) / len(rs), 4)
        # tuned Nox, for the comparison table
        ft = json.load(open(C.DATA / 'results' / 'e8' / 'ft-cascade.json'))
        for k in ('tuned-lenient', 'tuned-strict'):
            if f'{part}/{k}' in ft: OUT[f'{part}/nox-{k}'] = ft[f'{part}/{k}']
    for k, v in OUT.items():
        print(f'{k:28s}', ' '.join(f'{a}={b}' for a, b in v.items() if a not in ('n', 'big_ms')))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(OUT, open(out, 'w'), indent=1); print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else C.DATA / 'results' / 'e9' / 'vela-cascade.json')
