"""Every number in paper-cascade/ from the frozen results, the pre-registered gates and the live runs.

    python3 cascade_paper.py [--out ../../paper-cascade/gen]     # DATA=~/data/decision-eval/v1 (default)

Writes <out>/numbers.tex (one \\num macro per figure quoted in the text), <out>/*.dat (pgfplots data)
and <out>/numbers.json. Three blocks:

  sim     the analytical (replay) model: cascade.py over the frozen results, for every small model,
          with latency from single-stream service times (decision models: frozen, which run.py ran
          single-stream; qwen: the live single-stream run, because the frozen qwen run was 8-way).
  live    the Nox -> qwen cascade run for real by cascade_live.py, at concurrency 1 and 8.
  check   prediction vs measurement, row by row where possible.
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade as C

DATA = C.DATA; LIVE = DATA / 'cascade' / 'live'
ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'paper-cascade' / 'gen'))
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
PARTS = ['e3', 'e4_ent', 'e4_facts', 'e2']
SMALLS = ['nox', 'kai', 'clef-flash', 'clef', 'lux']
G = json.load(open(DATA / 'cascade' / 'gates-nox-qwen.json'))['gates']
NUM, J = {}, {}


def num(key, v, fmt='{}'):
    NUM[key] = fmt.format(v) if not isinstance(v, str) else v
    return v


mean = statistics.mean
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]
def live(part, mode, c, tag=''):
    p = LIVE / part / f'{mode}-c{c}{tag}.json'
    return json.load(open(p)) if p.exists() else None
def by_i(doc): return {r['i']: r for r in doc['rows'] if 'error' not in r}
def k(part): return part.replace('_', '')


# ---------------------------------------------------------------------------------------------
# ground truth per row (from the frozen nox/qwen pairing -- gold does not depend on the model)
GOLD = {part: {r['i']: r for r in C.rows(part, 'nox', 'qwen')} for part in PARTS}

# single-stream qwen service time per row, from the live c=1 run (falls back to frozen if absent)
QWEN1 = {}
for part in PARTS:
    d = live(part, 'big', 1)
    QWEN1[part] = {i: r['ms'] for i, r in by_i(d).items()} if d else None

# ---------------------------------------------------------------------------------------------
# SIM: design space over small models, with corrected (single-stream) latency
t0 = time.time(); n_evals = 0
SIM = {}
for part in PARTS:
    for sm in SMALLS:
        rs = C.rows(part, sm, 'qwen')
        if QWEN1[part]:
            rs = [dict(r, bms=QWEN1[part][r['i']]) for r in rs if r['i'] in QWEN1[part]]
        n = len(rs); ok_b = [r['bp'] == r['gold'] for r in rs]; ok_s = [r['sp'] == r['gold'] for r in rs]
        ok, esc, ms, th = C.heldout(rs, 0); n_evals += 2 * (n + 2)
        lo, hi = C.ci([int(x) - int(y) for x, y in zip(ok, ok_b)])
        row = dict(n=n, small_acc=100 * sum(ok_s) / n, big_acc=100 * sum(ok_b) / n, acc=100 * sum(ok) / n,
                   d_big=100 * (sum(ok) - sum(ok_b)) / n, ci=(100 * lo, 100 * hi), esc=100 * sum(esc) / n,
                   ms=mean(ms), small_ms=mean(r['sms'] for r in rs), big_ms=mean(r['bms'] for r in rs),
                   missed=C.missed(rs, ok) if part == 'e3' else None, big_missed=C.missed(rs, ok_b) if part == 'e3' else None)
        SIM[part, sm] = row
        # Pareto sweep (all rows, not held out) for the figure
        with open(OUT / f'sweep-{k(part)}-{sm}.dat', 'w') as f:
            f.write('t esc acc ms\n')
            for t in sorted({0.0} | {round(x / 100, 2) for x in range(30, 101, 2)} | {1.01}):
                o, e, m = C.run(rs, t); n_evals += n
                f.write(f'{t:.2f} {100 * sum(e) / n:.2f} {100 * sum(o) / n:.2f} {mean(m):.1f}\n')
sim_s = time.time() - t0
num('sim:seconds', sim_s, '{:.1f}'); num('sim:evals', f'{n_evals:,}'.replace(',', '{,}'))
J['sim'] = {f'{p}/{s}': v for (p, s), v in SIM.items()}
for (part, sm), r in SIM.items():
    key = f'sim:{k(part)}:{sm}'
    for f_, fmt in (('acc', '{:.1f}'), ('small_acc', '{:.1f}'), ('big_acc', '{:.1f}'), ('d_big', '{:+.1f}'),
                    ('esc', '{:.0f}'), ('ms', '{:.0f}'), ('small_ms', '{:.0f}'), ('big_ms', '{:.0f}')):
        num(f'{key}:{f_}', r[f_], fmt)
    num(f'{key}:ci', f'[{r["ci"][0]:+.1f}, {r["ci"][1]:+.1f}]')
    num(f'{key}:speedup', r['big_ms'] / r['ms'], '{:.1f}')

# design-space table
with open(OUT / 'tab-design.tex', 'w') as f:
    f.write('\\begin{tabular}{llrrrrrrr}\n\\toprule\n'
            'Task & Small & \\multicolumn{3}{c}{Accuracy (\\%)} & $\\Delta$ vs qwen (pt) & Esc. & Mean & Speed-up \\\\\n'
            '\\cmidrule(lr){3-5}\n & model & small & qwen & cascade & [95\\% CI] & (\\%) & (ms) & vs qwen \\\\\n\\midrule\n')
    for j, part in enumerate(('e3', 'e4_facts', 'e4_ent', 'e2')):
        name = {'e3': 'E3 triage', 'e4_facts': 'E4 fact typing', 'e4_ent': 'E4 entity typing', 'e2': 'E2 spelling'}[part]
        first = True
        for sm in SMALLS:
            r = SIM[part, sm]
            lab = f'\\multirow{{{len(SMALLS)}}}{{*}}{{{name}}}' if first else ''; first = False
            mis = f' ({r["missed"]}/{r["big_missed"]})' if part == 'e3' else ''
            f.write(f'{lab} & {sm} & {r["small_acc"]:.1f} & {r["big_acc"]:.1f} & {r["acc"]:.1f}{mis} & '
                    f'{r["d_big"]:+.1f} [{r["ci"][0]:+.1f}, {r["ci"][1]:+.1f}] & {r["esc"]:.0f} & {r["ms"]:.0f} & '
                    f'{r["big_ms"] / r["ms"]:.1f}$\\times$ \\\\\n')
        f.write('\\midrule\n' if j < 3 else '\\bottomrule\n\\end{tabular}\n')

# ---------------------------------------------------------------------------------------------
# LIVE + CHECK (Nox -> qwen)
def acc_of(part, preds):
    gold = GOLD[part]; return 100 * sum(preds[i] == gold[i]['gold'] for i in preds) / len(preds)

FROZEN_N = {part: {r['i']: r for r in C.rows(part, 'nox', 'qwen')} for part in PARTS}
CFGS = [('', 'mtp2', 'MTP-2, prefix cache on (deployed)'), ('-mtp0', 'mtp0', 'MTP off, prefix cache on'),
        ('-mtp0-apc0', 'mtp0apc0', 'MTP off, prefix cache off')]


def check(part, tag):
    """Live Nox -> qwen cascade under one qwen serving config (tag) vs the replay prediction.
    Nox runs are untagged: the small model's server does not change across configs."""
    s1, s8 = live(part, 'small', 1), live(part, 'small', 8)
    b1, c1, b8, c8 = (live(part, m, c, tag) for m, c in (('big', 1), ('cascade', 1), ('big', 8), ('cascade', 8)))
    if not (s1 and b1 and c1): return None
    fr = FROZEN_N[part]; g = G[part]
    S1, B1, C1 = by_i(s1), by_i(b1), by_i(c1)
    ids = sorted(set(S1) & set(B1) & set(C1) & set(fr)); n = len(ids)
    r = dict(n=n)
    # 1. reproducibility of each model against the frozen run
    r['small_agree'] = 100 * sum(S1[i]['pred'] == fr[i]['sp'] for i in ids) / n
    r['small_dconf'] = max(abs(S1[i]['conf'] - fr[i]['sc']) for i in ids)
    r['big1_agree'] = 100 * sum(B1[i]['pred'] == fr[i]['bp'] for i in ids) / n
    if b8:
        B8 = by_i(b8); r['big8_agree'] = 100 * sum(B8[i]['pred'] == fr[i]['bp'] for i in ids if i in B8) / n
        r['big1v8_agree'] = 100 * sum(B8[i]['pred'] == B1[i]['pred'] for i in ids if i in B8) / n
    # 2. accuracy & escalation: predicted (gates) vs live
    r['sim_acc'] = acc_of(part, {i: g[str(i)]['pred'] for i in ids}); r['sim_esc'] = 100 * sum(g[str(i)]['esc'] for i in ids) / n
    r['live_acc'] = acc_of(part, {i: C1[i]['pred'] for i in ids}); r['live_esc'] = 100 * sum(C1[i]['esc'] for i in ids) / n
    r['big_live_acc'] = acc_of(part, {i: B1[i]['pred'] for i in ids}); r['small_live_acc'] = acc_of(part, {i: S1[i]['pred'] for i in ids})
    r['big_frozen_acc'] = acc_of(part, {i: fr[i]['bp'] for i in ids}); r['small_frozen_acc'] = acc_of(part, {i: fr[i]['sp'] for i in ids})
    r['esc_agree'] = 100 * sum(C1[i]['esc'] == g[str(i)]['esc'] for i in ids) / n
    r['pred_agree'] = 100 * sum(C1[i]['pred'] == g[str(i)]['pred'] for i in ids) / n
    okc = [C1[i]['pred'] == GOLD[part][i]['gold'] for i in ids]; okb = [B1[i]['pred'] == GOLD[part][i]['gold'] for i in ids]
    lo, hi = C.ci([int(x) - int(y) for x, y in zip(okc, okb)]); r['live_d_big'] = 100 * (sum(okc) - sum(okb)) / n; r['live_ci'] = (100 * lo, 100 * hi)
    if part == 'e3':
        rs = [GOLD[part][i] for i in ids]
        r['live_missed'] = C.missed(rs, okc); r['big_live_missed'] = C.missed(rs, okb)
        r['sim_missed'] = C.missed(rs, [g[str(i)]['pred'] == GOLD[part][i]['gold'] for i in ids])
    if c8:
        C8 = by_i(c8); r['live8_acc'] = acc_of(part, {i: C8[i]['pred'] for i in ids if i in C8})
        r['live8_esc'] = 100 * sum(C8[i]['esc'] for i in ids if i in C8) / n
    # 3. latency at c=1
    meas = [C1[i]['ms'] for i in ids]
    naive = [fr[i]['sms'] + (fr[i]['bms'] if g[str(i)]['esc'] else 0) for i in ids]             # frozen ms (qwen 8-way)
    comp = [S1[i]['ms'] + (B1[i]['ms'] if g[str(i)]['esc'] else 0) for i in ids]                 # live c=1 service times
    comp_live = [S1[i]['ms'] + (B1[i]['ms'] if C1[i]['esc'] else 0) for i in ids]
    for nm, xs in (('meas', meas), ('naive', naive), ('comp', comp), ('complive', comp_live),
                   ('small', [S1[i]['ms'] for i in ids]), ('big', [B1[i]['ms'] for i in ids])):
        r[f'{nm}_mean'], r[f'{nm}_p50'], r[f'{nm}_p95'] = mean(xs), pct(xs, .5), pct(xs, .95)
    r['overhead_ms'] = mean(m - c for m, c in zip(meas, comp_live))
    # the cascade's own stage timings: drift of each model's service time between runs vs router cost
    r['small_in_casc_mean'] = mean(C1[i]['small_ms'] for i in ids)
    esc_ids = [i for i in ids if C1[i]['esc']]
    if esc_ids:
        r['big_in_casc_mean'] = mean(C1[i]['big_ms'] for i in esc_ids); r['big_alone_esc_mean'] = mean(B1[i]['ms'] for i in esc_ids)
    r['router_ms'] = mean(C1[i]['ms'] - C1[i]['small_ms'] - C1[i].get('big_ms', 0) for i in ids)
    r['speedup_meas'] = r['big_mean'] / r['meas_mean']; r['speedup_pred'] = r['big_mean'] / r['comp_mean']
    # 4. throughput at c=8: measured vs a load-dependent closed-network model
    if s8 and b8 and c8:
        c = 8; p = r['sim_esc'] / 100
        Rs = lambda x: r['small_mean'] + (mean(q['ms'] for q in s8['rows'] if 'error' not in q) - r['small_mean']) * (x - 1) / (c - 1)
        Rb = lambda x: r['big_mean'] + (mean(q['ms'] for q in b8['rows'] if 'error' not in q) - r['big_mean']) * (x - 1) / (c - 1)
        def jobs(X):  # clients a throughput X implies, solving n = X R(n) at each station by fixed point
            ns = nb = 1.0
            for _ in range(200):
                ns = max(1.0, X * Rs(ns) / 1000); nb = max(1.0, X * p * Rb(nb) / 1000) if p else 0.0
            return ns + nb
        lo_, hi_ = 0.0, 1000.0
        for _ in range(60):
            mid = (lo_ + hi_) / 2; lo_, hi_ = (mid, hi_) if jobs(mid) < c else (lo_, mid)
        r['x_model'] = lo_
        r['x_naive'] = c * 1000 / r['comp_mean']                                         # no contention
        x_b8 = b8['n'] / b8['wall_s']; x_s8 = s8['n'] / s8['wall_s']
        r['x_bound'] = min(x_s8, x_b8 / p) if p else x_s8                                 # bottleneck bound
        r['x_meas'] = c8['n'] / c8['wall_s']; r['x_big8'] = x_b8; r['x_small8'] = x_s8
        r['tput_gain'] = r['x_meas'] / x_b8
        r['c8_mean'] = mean(q['ms'] for q in c8['rows'] if 'error' not in q); r['b8_mean'] = mean(q['ms'] for q in b8['rows'] if 'error' not in q)
    rep = live(part, 'big', 1, tag + '-rep')
    if rep:
        R = by_i(rep); r['big1_rep_agree'] = 100 * sum(R[i]['pred'] == B1[i]['pred'] for i in ids if i in R) / n
        r['big1_rep_mean'] = mean(R[i]['ms'] for i in ids if i in R)
        r['rep_slowdown'] = 100 * (r['big1_rep_mean'] - r['big_mean']) / r['big_mean']
    return r


def emit_chk(K, r):
    for f_, v in r.items():
        if isinstance(v, tuple): num(f'{K}:{f_}', f'[{v[0]:+.1f}, {v[1]:+.1f}]')
        elif isinstance(v, float):
            fmt = ('{:.2f}' if f_.startswith('x_') or f_ in ('small_dconf', 'router_ms') else
                   '{:.0f}' if f_.endswith(('_mean', '_p50', '_p95', '_ms')) else
                   '{:.1f}' if any(w in f_ for w in ('agree', 'acc', 'esc', 'speedup', 'd_big', 'gain')) else '{:.0f}')
            num(f'{K}:{f_}', v, fmt)
        else: num(f'{K}:{f_}', v)


CHK = {}
for tag, cfg, _ in CFGS:
    for part in PARTS:
        r = check(part, tag)
        if r: CHK[cfg, part] = r; emit_chk(f'chk:{cfg}:{k(part)}', r)
# qwen answers across serving configs (single-stream), row by row
for part in PARTS:
    runs = {cfg: live(part, 'big', 1, tag) for tag, cfg, _ in CFGS}
    for (a1, d1), (a2, d2) in ((x, y) for i, x in enumerate(runs.items()) for y in list(runs.items())[i + 1:]):
        if d1 and d2:
            X, Y = by_i(d1), by_i(d2); ids = sorted(set(X) & set(Y))
            num(f'xcfg:{k(part)}:{a1}:{a2}', 100 * sum(X[i]['pred'] == Y[i]['pred'] for i in ids) / len(ids), '{:.1f}')
J['check'] = {f'{c}/{p}': v for (c, p), v in CHK.items()}

NAME = {'e3': 'E3 triage', 'e4_facts': 'E4 fact typing', 'e4_ent': 'E4 entity typing', 'e2': 'E2 spelling'}
ORDER = ('e3', 'e4_facts', 'e4_ent', 'e2')
def tab(path, head, cols, rowfn, cfg='mtp2'):
    with open(OUT / path, 'w') as f:
        f.write(f'\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\\midrule\n')
        for part in ORDER:
            r = CHK.get((cfg, part))
            if r: f.write(f'{NAME[part]} & ' + ' & '.join(rowfn(r, part)) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

# live results: accuracy and speed of the real cascade (deployed qwen config)
tab('tab-live.tex',
    ' & \\multicolumn{4}{c}{Accuracy (\\%), $c=1$} & Esc. & \\multicolumn{3}{c}{Mean latency, $c=1$} & \\multicolumn{3}{c}{Throughput, $c=8$ (items/s)} \\\\\n'
    '\\cmidrule(lr){2-5}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n'
    'Task & Nox & qwen & cascade & $\\Delta$ [95\\% CI] & (\\%) & qwen & cascade & speed-up & qwen & cascade & gain \\\\\n',
    'lrrrrrrrrrrr',
    lambda r, part: [f"{r['small_live_acc']:.1f}", f"{r['big_live_acc']:.1f}", f"{r['live_acc']:.1f}",
                     f"{r['live_d_big']:+.1f} [{r['live_ci'][0]:+.1f}, {r['live_ci'][1]:+.1f}]", f"{r['live_esc']:.0f}",
                     f"{r['big_mean']:.0f}", f"{r['meas_mean']:.0f}", f"{r['speedup_meas']:.2f}$\\times$",
                     f"{r['x_big8']:.2f}", f"{r['x_meas']:.2f}", f"{r['tput_gain']:.2f}$\\times$"])
# serving-config ablation: one row per (config, task with escalation)
with open(OUT / 'tab-ablation.tex', 'w') as f:
    f.write('\\begin{tabular}{llrrrrrrrrr}\n\\toprule\n'
            ' & & \\multicolumn{4}{c}{qwen alone} & \\multicolumn{2}{c}{$\\Delta$ vs qwen (pt)} & \\multicolumn{2}{c}{Speed-up $c=1$} & Gain \\\\\n'
            '\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n'
            'qwen config & Task & ms & repro & $c1{=}c8$ & $c1{=}c1$ & pred. & meas. & pred. & meas. & $c=8$ \\\\\n\\midrule\n')
    have = [(tag, cfg, label) for tag, cfg, label in CFGS if any((cfg, p_) in CHK for p_ in ('e3', 'e4_facts', 'e2'))]
    for j, (tag, cfg, label) in enumerate(have):
        rows_ = [(part, CHK[cfg, part]) for part in ('e3', 'e4_facts', 'e2') if (cfg, part) in CHK]
        for m, (part, r) in enumerate(rows_):
            lab = f'\\multirow{{{len(rows_)}}}{{*}}{{\\shortstack[l]{{{label.replace(", ", ",\\\\")}}}}}' if m == 0 else ''
            g = lambda x, fmt: fmt.format(r[x]) if x in r else '--'
            f.write(f"{lab} & { {'e3': 'E3', 'e4_facts': 'E4 facts', 'e2': 'E2'}[part]} & {r['big_mean']:.0f} & {r['big1_agree']:.1f} & {g('big1v8_agree', '{:.1f}')} & "
                    f"{g('big1_rep_agree', '{:.1f}')} & {r['sim_acc'] - r['big_frozen_acc']:+.1f} & {r['live_d_big']:+.1f} & "
                    f"{r['speedup_pred']:.2f} & {r['speedup_meas']:.2f} & {g('tput_gain', '{:.2f}')} \\\\\n")
        f.write('\\midrule\n' if j < len(have) - 1 else '\\bottomrule\n\\end{tabular}\n')

# prediction vs measurement
def pe(pred, meas): return f'{100 * (pred - meas) / meas:+.0f}\\%'
tab('tab-check.tex',
    ' & \\multicolumn{2}{c}{Escalation} & \\multicolumn{2}{c}{Answers} & \\multicolumn{2}{c}{$\\Delta$ acc.\\ vs qwen (pt)} & \\multicolumn{4}{c}{Mean latency, $c=1$ (ms)} \\\\\n'
    '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-11}\n'
    'Task & pred. & agree & agree & qwen repro & pred. & meas. & meas. & naive & replay & error \\\\\n',
    'lrrrrrrrrrrr'[:-1],
    lambda r, part: [f"{r['sim_esc']:.1f}\\%", f"{r['esc_agree']:.0f}\\%", f"{r['pred_agree']:.1f}\\%", f"{r['big1_agree']:.1f}\\%",
                     f"{r['sim_acc'] - r['big_frozen_acc']:+.1f}", f"{r['live_d_big']:+.1f}",
                     f"{r['meas_mean']:.0f}", f"{r['naive_mean']:.0f}", f"{r['comp_mean']:.0f}", pe(r['comp_mean'], r['meas_mean'])])
tab('tab-tput.tex',
    ' & & \\multicolumn{4}{c}{Cascade throughput, $c=8$ (items/s)} & \\multicolumn{3}{c}{Prediction error} \\\\\n'
    '\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\n'
    'Task & Esc. & no contention & bound & model & measured & no cont. & bound & model \\\\\n',
    'lrrrrrrrr',
    lambda r, part: [f"{r['sim_esc']:.0f}\\%", f"{r['x_naive']:.1f}", f"{r['x_bound']:.2f}", f"{r['x_model']:.2f}", f"{r['x_meas']:.2f}",
                     pe(r['x_naive'], r['x_meas']), pe(r['x_bound'], r['x_meas']), pe(r['x_model'], r['x_meas'])])

with open(OUT / 'numbers.tex', 'w') as f:
    f.write('% generated by clef/eval/cascade_paper.py -- do not edit\n')
    for key, v in sorted(NUM.items()):
        f.write(f'\\expandafter\\def\\csname num:{key}\\endcsname{{{v}}}\n')
json.dump(J, open(OUT / 'numbers.json', 'w'), indent=1, default=str)
print(f'{len(NUM)} numbers, sim {sim_s:.1f}s; live parts checked: {list(CHK)}')
for part, r in CHK.items():
    print(part, {x: (round(v, 2) if isinstance(v, float) else v) for x, v in r.items()})
