"""Every number, table and figure in paper-specialize/ (the tuned-first-stage cascade paper).

    python3 ft_paper.py [--out ../../paper-specialize/gen]

Reads:
  results/e8/ft-cascade.json        replay: released vs tuned Nox -> qwen (cascade_ft.py score)
  results/e8/prereg.json            frozen gates + head hashes + predictions, written before the live run
  cascade/live/<part>/*-ft-*.json   the live run (cascade_ft_live.sh): tuned Nox alone and the cascade per
                                    half, c=1 and c=8; qwen alone on every row in the same session (-ft)
  paper-cascade/gen/numbers.tex     the released cascade's live figures, for the before/after comparison
Writes <out>/numbers.tex (\\num macros), <out>/numbers.json and the tables.
"""
import argparse, collections, hashlib, json, re, statistics
from pathlib import Path
import cascade as C, cascade_ft as X

ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'paper-specialize' / 'gen'))
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
E8 = C.DATA / 'results' / 'e8'; LIVE = C.DATA / 'cascade' / 'live'
VAR = {'e3': 'triage-lenient', 'e4_facts': 'typing-strict'}
K = {'e3': 'e3', 'e4_facts': 'e4facts'}
NUM, J = {}, {}
mean = statistics.mean


def num(key, v, fmt='{}'):
    NUM[key] = v if isinstance(v, str) else fmt.format(v); return v


def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


# ---------------------------------------------------------------------------------------------
# REPLAY: released / tuned (lenient = groups as the replay; strict = also entity/token-disjoint)
R = json.load(open(E8 / 'ft-cascade.json')); J['replay'] = R
for key, r in R.items():
    part, var = key.split('/'); k = f'rep:{K[part]}:{var.replace("tuned-", "")}'
    for f_, fmt in (('small_acc', '{:.1f}'), ('big_acc', '{:.1f}'), ('cascade_acc', '{:.1f}'), ('d_big', '{:+.1f}'),
                    ('esc', '{:.0f}'), ('mean_ms', '{:.0f}'), ('big_ms', '{:.0f}'), ('speedup', '{:.2f}')):
        num(f'{k}:{f_}', r[f_], fmt)
    num(f'{k}:ci', f'[{r["ci"][0]:+.1f}, {r["ci"][1]:+.1f}]')
    if part == 'e3':
        for f_ in ('missed', 'big_missed', 'small_missed'): num(f'{k}:{f_}', r[f_])

# E4: labels are entity-level; a lookup of the other half's rulings
f4 = json.load(open(C.DATA / 'e4_facts.json')); rs4 = X.base_rows('e4_facts')
ent = collections.defaultdict(collections.Counter)
for x in f4: ent[x['key']][x['gold']] += 1
num('e4:consistent', 100 * sum(ent[x['key']].most_common(1)[0][0] == x['gold'] for x in f4) / len(f4), '{:.0f}')
num('e4:n_entities', len(ent)); num('e4:n_chapters', len({x['chapter'] for x in f4}))
ok = cov = n = 0
for tr, te in C.folds(rs4):
    maj = collections.defaultdict(collections.Counter)
    for r in rs4:
        if r['g'] in tr: maj[f4[r['i']]['key']][r['gold']] += 1
    for r in rs4:
        if r['g'] not in te: continue
        n += 1; m = maj[f4[r['i']]['key']]
        p = m.most_common(1)[0][0] if m else r['sp']; cov += bool(m); ok += p == r['gold']
num('e4:lookup_acc', 100 * ok / n, '{:.1f}'); num('e4:lookup_cov', 100 * cov / n, '{:.0f}')

# how much training data the strict variants keep
for task, part in (('triage', 'e3'), ('typing', 'e4_facts')):
    for mode in ('lenient', 'strict'):
        F = load(E8 / f'{task}-{mode}.json')
        if F:
            num(f'ntrain:{K[part]}:{mode}', ' / '.join(str(F['halves'][h]['n_train']) for h in ('0', '1')))
P3 = json.load(open(E8 / 'plan-triage.json'))['rows']; T3 = json.load(open(C.DATA / 'e6_triage_items.json'))
rec = 0; names = 0
for h in (0, 1):
    trk = {p['key'] for p in P3 if p['half'] != h}
    for i, p in enumerate(P3):
        if p['half'] == h and T3[i]['label'] == 'name': names += 1; rec += p['key'] in trk
num('e3:names_recur', rec); num('e3:names', names)

REG = json.load(open(E8 / 'prereg.json'))
num('prereg:sha', hashlib.sha256(open(E8 / 'prereg.json', 'rb').read()).hexdigest()[:8])
num('prereg:when', REG['written'])

# ---------------------------------------------------------------------------------------------
# LIVE
GOLD = {part: {r['i']: r for r in X.base_rows(part)} for part in VAR}
PRED = {part: {} for part in VAR}   # frozen per-row gate + prediction
for part, var in VAR.items():
    for h in (0, 1):
        for i, g in json.load(open(E8 / f'live-gates-{part}-{var}-h{h}.json'))['gates'][part].items():
            PRED[part][int(i)] = dict(g, half=h)
REPLAY_P = {part: {} for part in VAR}
for part, var in VAR.items():
    F = json.load(open(E8 / f'{var}.json'))
    for h in ('0', '1'):
        for i, p in F['halves'][h]['test'].items(): REPLAY_P[part][int(i)] = X.small(part, p)[1]


def rows_of(doc): return {r['i']: r for r in doc['rows'] if 'error' not in r}


def ci_diff(a_, b_):
    lo, hi = C.ci([int(x) - int(y) for x, y in zip(a_, b_)]); return 100 * lo, 100 * hi


LV = {}
for part, var in VAR.items():
    k = K[part]; gold = GOLD[part]
    halves = {h: {f'{m}{c}': load(LIVE / part / f'{m}-c{c}-ft-{var}-h{h}.json') for m in ('small', 'cascade') for c in (1, 8)}
              for h in (0, 1)}
    b1, b8 = load(LIVE / part / 'big-c1-ft.json'), load(LIVE / part / 'big-c8-ft.json')
    if not (b1 and all(halves[h]['cascade1'] for h in (0, 1))):
        print('live data missing for', part); continue
    B1 = rows_of(b1)
    S1 = {**rows_of(halves[0]['small1']), **rows_of(halves[1]['small1'])}
    C1 = {**rows_of(halves[0]['cascade1']), **rows_of(halves[1]['cascade1'])}
    idx = sorted(i for i in C1 if i in B1 and i in S1 and i in gold)
    g_ = lambda i: gold[i]['gold']
    ok_c = [C1[i]['pred'] == g_(i) for i in idx]; ok_b = [B1[i]['pred'] == g_(i) for i in idx]
    ok_s = [S1[i]['pred'] == g_(i) for i in idx]
    rs = [gold[i] for i in idx]
    r = dict(n=len(idx), errors=sum(d['errors'] for h in (0, 1) for d in halves[h].values() if d) + b1['errors'],
             live_acc=100 * mean(ok_c), big_live_acc=100 * mean(ok_b), small_live_acc=100 * mean(ok_s),
             d_big=100 * (sum(ok_c) - sum(ok_b)) / len(idx))
    r['ci'] = ci_diff(ok_c, ok_b)
    r['esc'] = 100 * mean(C1[i]['esc'] for i in idx)
    r['esc_agree'] = 100 * mean(C1[i]['esc'] == PRED[part][i]['esc'] for i in idx)
    r['conf_maxdiff'] = max(abs(C1[i]['conf'] - REPLAY_P[part][i]) for i in idx)
    # answers: cascade vs pre-registered prediction; qwen live vs its own frozen answer
    r['ans_agree_all'] = 100 * mean(C1[i]['pred'] == (gold[i]['bp'] if PRED[part][i]['esc'] else PRED[part][i]['pred']) for i in idx)
    r['big_agree'] = 100 * mean(B1[i]['pred'] == gold[i]['bp'] for i in idx)
    if part == 'e3':
        r['live_missed'] = C.missed(rs, ok_c); r['big_live_missed'] = C.missed(rs, ok_b); r['small_live_missed'] = C.missed(rs, ok_s)
    # latency, c=1
    cm = [C1[i]['ms'] for i in idx]; bm = [B1[i]['ms'] for i in idx]; sm = [S1[i]['ms'] for i in idx]
    r.update(meas_mean=mean(cm), big_mean=mean(bm), small_mean=mean(sm), meas_p50=pct(cm, .5), meas_p95=pct(cm, .95),
             big_p95=pct(bm, .95), speedup_meas=mean(bm) / mean(cm))
    # replay latency composed from this session's single-stream service times
    comp = [S1[i]['ms'] + (B1[i]['ms'] if PRED[part][i]['esc'] else 0) for i in idx]
    r['comp_mean'] = mean(comp); r['speedup_pred'] = mean(bm) / mean(comp)
    r['lat_err'] = 100 * (r['comp_mean'] - r['meas_mean']) / r['meas_mean']
    # throughput, c=8: the two halves run back to back, so pool rows / pooled wall time
    if b8 and all(halves[h]['cascade8'] for h in (0, 1)):
        x_c = sum(halves[h]['cascade8']['n'] for h in (0, 1)) / sum(halves[h]['cascade8']['wall_s'] for h in (0, 1))
        x_s = sum(halves[h]['small8']['n'] for h in (0, 1)) / sum(halves[h]['small8']['wall_s'] for h in (0, 1))
        x_b = b8['n'] / b8['wall_s']
        r.update(tput_cascade=x_c, tput_small=x_s, tput_big=x_b, tput_gain=x_c / x_b)
    LV[part] = r
    for f_, v in r.items():
        if isinstance(v, tuple): num(f'live:{k}:{f_}', f'[{v[0]:+.1f}, {v[1]:+.1f}]'); continue
        fmt = ('{:+.1f}' if f_ in ('d_big', 'lat_err') else '{:.4f}' if f_ == 'conf_maxdiff' else
               '{:.2f}' if f_.startswith('speedup') or f_.startswith('tput') else
               '{:.1f}' if any(w in f_ for w in ('acc', 'agree', 'esc')) else '{:.0f}')
        num(f'live:{k}:{f_}', v, fmt)
J['live'] = LV

# released cascade, live, from the cascade paper (same protocol, a day earlier)
PC = {}
for line in open(Path(__file__).resolve().parents[2] / 'paper-cascade' / 'gen' / 'numbers.tex'):
    m = re.match(r'\\expandafter\\def\\csname num:(chk:mtp2:[^\\]+)\\endcsname\{(.*)\}$', line.strip())
    if m: PC[m.group(1)] = m.group(2)
for k in ('e3', 'e4facts'):
    for f_ in ('live_acc', 'big_live_acc', 'live_missed', 'big_live_missed', 'speedup_meas', 'tput_gain', 'meas_mean',
               'big_mean', 'sim_esc'):
        if f'chk:mtp2:{k}:{f_}' in PC: num(f'rel:{k}:{f_}', PC[f'chk:mtp2:{k}:{f_}'])

# ---------------------------------------------------------------------------------------------
# TUNING OUTCOME, in Vela 2.0's metrics: released vs tuned Nox alone, from the cross-fitted held-out
# probabilities (each item scored by a head that never saw its label). E3 binary AUC (campaign name or
# not) + accuracy; E4 fact typing macro-F1 + accuracy.
def auc(pos, neg):
    neg = sorted(neg); import bisect
    return sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) for x in pos) / len(pos) / len(neg)

def macro_f1(gold, pred):
    f = []
    for c in sorted(set(gold)):
        tp = sum(g == c and q == c for g, q in zip(gold, pred)); fp = sum(g != c and q == c for g, q in zip(gold, pred))
        fn = sum(g == c and q != c for g, q in zip(gold, pred))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / len(f)

GAIN = {}
for part, task in X.PARTS.items():
    rs = X.base_rows(part); S = C.load(part, 'nox'); ids = [r['i'] for r in rs]; gold = {r['i']: r['gold'] for r in rs}
    probs = {'released': {i: S[i]['p'] for i in ids}}
    for mode in ('lenient', 'strict'):
        F_ = load(E8 / f'{task}-{mode}.json')
        if F_: probs[mode] = {int(i): p for h in F_['halves'].values() for i, p in h['test'].items()}
    for v, P in probs.items():
        if part == 'e3':
            sc = {i: P[i]['campaign'] for i in ids}
            m = dict(auc=auc([sc[i] for i in ids if gold[i]], [sc[i] for i in ids if not gold[i]]),
                     acc=sum((max(P[i], key=P[i].get) == 'campaign') == gold[i] for i in ids) / len(ids))
        else:
            pred = [max(P[i], key=P[i].get) for i in ids]
            m = dict(f1=macro_f1([gold[i] for i in ids], pred), acc=sum(p_ == gold[i] for p_, i in zip(pred, ids)) / len(ids))
        GAIN[f'{part}/{v}'] = m
        for f_, x in m.items(): num(f'gain:{K[part]}:{v}:{f_}', 100 * x if f_ == 'acc' else x, '{:.1f}' if f_ == 'acc' else '{:.3f}')
J['gain'] = GAIN

# Vela 2.0 4B's published in-domain results (EVALUATION.md, vllm-sr/Vela-2.0-4B @ d692394): task, metric,
# reference model, reference score, Vela 4B score, delta points [95% CI] as published. Not re-measured here.
VELA_PUB = [
    ('Prompt attacks, unseen families', 'AUC', 'Guard', 0.792, 0.985, '+19.4 [+15.1, +23.7]'),
    ('Hate speech, Multilingual HateCheck', 'AUC', 'Safety', 0.646, 0.769, '+12.3 [+11.7, +12.9]'),
    ('Request harm, CultureGuard', 'AUC', 'Safety', 0.863, 0.898, '+3.5 [+3.2, +3.8]'),
    ('Request harm, RTP-LX', 'AUC', 'Safety', 0.761, 0.774, '+1.3 [+0.8, +1.8]'),
    ('Fact-check needed', 'macro-F1', 'FactCheck', 0.501, 1.000, '+49.9 [+46.1, +53.5]'),
    ('Feedback', 'macro-F1', 'Feedback', 0.425, 0.920, '+49.5 [+47.0, +52.2]'),
    ('Modality', 'macro-F1', 'Modality', 0.889, 1.000, '+11.1 [+9.4, +12.7]'),
    ('Domain', 'macro-F1', 'Domain', 0.831, 0.820, '$-$1.2 [$-$2.8, +0.5]'),
    ('PII, short texts', 'F1', 'PII', 0.976, 0.987, '+1.1 [+0.2, +2.1]'),
    ('PII, 8K-token documents', 'F1', 'PII', 0.908, 0.905, '$-$0.3 [$-$3.6, +3.3]'),
    ('Hallucination spans', 'F1', 'Halu', 0.875, 0.874, '$-$0.1 [$-$0.7, +0.6]'),
    ('RAGTruth', 'F1', 'LettuceDetect v2 mmBERT', 0.743, 0.770, '+2.7 [+0.4, +5.1]'),
    ('Toxic spans', 'char-F1', 'mmBERT span head', 0.451, 0.379, '$-$7.2 [$-$8.0, $-$6.5]'),
]
num('vela:jev', '31.91'); num('vela:jev_nox', '42.55')
num('vela:n_up', sum(1 for r in VELA_PUB if not r[5].startswith('$')))
num('vela:n_tasks', len(VELA_PUB))

# Vela 2.0 members as the cascade's first stage on OUR tasks (vela_replay.py), replay only
VR = load(C.DATA / 'results' / 'e9' / 'vela-cascade.json'); J['vela_first'] = VR
if VR:
    for key, r in VR.items():
        part, m = key.split('/'); k = f'vf:{K[part]}:{m}'
        for f_, fmt in (('small_acc', '{:.1f}'), ('cascade_acc', '{:.1f}'), ('d_big', '{:+.1f}'), ('esc', '{:.0f}'),
                        ('mean_ms', '{:.0f}'), ('speedup', '{:.2f}')):
            num(f'{k}:{f_}', r[f_], fmt)
        num(f'{k}:ci', f'[{r["ci"][0]:+.1f}, {r["ci"][1]:+.1f}]')
    VSM = {}
    for part in ('e3', 'e4_facts'):
        for m in ('nox', 'vela-0.8b', 'vela-4b', 'vela-9b'):
            try: VSM[(part, m)] = mean(r['sms'] for r in X.base_rows(part) if True) if m == 'nox' else \
                mean(x['ms'] for x in C.load(part, m) if 'error' not in x)
            except FileNotFoundError: pass
    for (part, m), v in VSM.items(): num(f'vf:{K[part]}:{m}:small_ms', v, '{:.0f}')

# ---------------------------------------------------------------------------------------------
# tables
with open(OUT / 'tab-replay.tex', 'w') as f:
    f.write('\\begin{tabular}{llrrrrrr}\n\\toprule\n'
            'Task & First stage & Nox alone & Cascade & $\\Delta$ vs qwen (pt) [95\\% CI] & Esc.\\ (\\%) & Missed & Speed-up \\\\\n\\midrule\n')
    lab = {'released': 'released', 'tuned-lenient': 'tuned', 'tuned-strict': 'tuned, disjoint'}
    for part, nm in (('e3', 'E3 triage'), ('e4_facts', 'E4 fact typing')):
        first = True
        for v in ('released', 'tuned-lenient', 'tuned-strict'):
            r = R.get(f'{part}/{v}')
            if not r: continue
            mis = f"{r['missed']} / {r['big_missed']}" if part == 'e3' else '--'
            f.write(f"{nm if first else ''} & {lab[v]} & {r['small_acc']:.1f} & {r['cascade_acc']:.1f} & "
                    f"{r['d_big']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] & {r['esc']:.0f} & {mis} & {r['speedup']:.2f}$\\times$ \\\\\n")
            first = False
        f.write('\\midrule\n' if part == 'e3' else '')
    f.write(f"E4 fact typing & entity lookup & \\multicolumn{{2}}{{r}}{{{NUM['e4:lookup_acc']} (coverage {NUM['e4:lookup_cov']}\\%)}} & & & & \\\\\n")
    f.write('\\bottomrule\n\\end{tabular}\n')

if LV:
    with open(OUT / 'tab-live.tex', 'w') as f:
        f.write('\\begin{tabular}{llrrrrrrr}\n\\toprule\n'
                'Task & First stage & Cascade & qwen & $\\Delta$ (pt) & Esc. & Missed & Speed-up & Throughput \\\\\n'
                ' & & acc.\\ (\\%) & acc.\\ (\\%) & [95\\% CI] & (\\%) & (casc./qwen) & $c{=}1$ & gain $c{=}8$ \\\\\n\\midrule\n')
        for part, nm in (('e3', 'E3 triage'), ('e4_facts', 'E4 fact typing')):
            k = K[part]; g = lambda f_: NUM.get(f'rel:{k}:{f_}', '--')
            mis_rel = f"{g('live_missed')} / {g('big_live_missed')}" if part == 'e3' else '--'
            f.write(f"{nm} & released$^\\dagger$ & {g('live_acc')} & {g('big_live_acc')} & & {float(g('sim_esc')):.0f} & {mis_rel} & "
                    f"{g('speedup_meas')}$\\times$ & {g('tput_gain')}$\\times$ \\\\\n")
            r = LV.get(part)
            if r:
                mis = f"{r['live_missed']} / {r['big_live_missed']}" if part == 'e3' else '--'
                f.write(f" & tuned & {r['live_acc']:.1f} & {r['big_live_acc']:.1f} & {r['d_big']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] & "
                        f"{r['esc']:.0f} & {mis} & {r['speedup_meas']:.2f}$\\times$ & {r.get('tput_gain', float('nan')):.2f}$\\times$ \\\\\n")
            f.write('\\midrule\n' if part == 'e3' else '')
        f.write('\\bottomrule\n\\end{tabular}\n')

    with open(OUT / 'tab-check.tex', 'w') as f:
        f.write('\\begin{tabular}{lrrrrrrrr}\n\\toprule\n'
                'Task & Esc.\\ pred. & Esc.\\ agree & max $|\\Delta p|$ & Answers agree & qwen self-agree & $\\Delta$ pred. & $\\Delta$ meas. & Latency err. \\\\\n\\midrule\n')
        for part, nm in (('e3', 'E3 triage'), ('e4_facts', 'E4 fact typing')):
            r = LV.get(part)
            if not r: continue
            p = REG[f'{part}/predicted']
            f.write(f"{nm} & {p['esc']:.1f}\\% & {r['esc_agree']:.1f}\\% & {r['conf_maxdiff']:.4f} & {r['ans_agree_all']:.1f}\\% & "
                    f"{r['big_agree']:.1f}\\% & {p['d_big']:+.1f} & {r['d_big']:+.1f} & {r['lat_err']:+.0f}\\% \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')

with open(OUT / 'tab-gains.tex', 'w') as f:
    f.write('\\begin{tabular}{lllrrl}\n\\toprule\n'
            'Model & Task (held out) & Metric & Before & After & $\\Delta$ (pt) \\\\\n\\midrule\n')
    for part, nm, met in (('e3', 'E3 triage, by session', 'auc'), ('e3', 'E3 triage, by session', 'acc'),
                          ('e4_facts', 'E4 fact typing, by entity', 'f1'), ('e4_facts', 'E4 fact typing, by entity', 'acc'),
                          ('e4_facts', 'E4 fact typing, by chapter', 'f1')):
        v = 'strict' if 'entity' in nm else 'lenient'
        a_, b_ = GAIN[f'{part}/released'][met], GAIN[f'{part}/{v}'][met]
        sc = 100 if met == 'acc' else 1; fm = '{:.1f}' if met == 'acc' else '{:.3f}'
        lab = {'auc': 'AUC', 'acc': 'accuracy (\\%)', 'f1': 'macro-F1'}[met]
        f.write(f"Nox-4B, tuned head & {nm} & {lab} & {fm.format(sc * a_)} & {fm.format(sc * b_)} & {100 * (b_ - a_):+.1f} \\\\\n")
    f.write('\\midrule\n')
    for t, met, ref, rs_, vs, d in VELA_PUB:
        f.write(f"Vela 2.0 4B$^\\ddagger$ & {t} & {met} & {rs_:.3f} & {vs:.3f} & {d} \\\\\n")
    f.write('\\bottomrule\n\\end{tabular}\n')

if VR:
    with open(OUT / 'tab-vela-first.tex', 'w') as f:
        f.write('\\begin{tabular}{llrrrrrr}\n\\toprule\n'
                'Task & First stage & Alone (\\%) & Cascade (\\%) & $\\Delta$ vs qwen (pt) [95\\% CI] & Esc.\\ (\\%) & ms/item & Speed-up \\\\\n\\midrule\n')
        lab = {'nox': 'Nox 4B, released', 'nox-tuned-lenient': 'Nox 4B, tuned', 'nox-tuned-strict': 'Nox 4B, tuned, disjoint',
               'vela-0.8b': 'Vela 2.0 0.8B', 'vela-4b': 'Vela 2.0 4B', 'vela-9b': 'Vela 2.0 9B'}
        for part, nm in (('e3', 'E3 triage'), ('e4_facts', 'E4 fact typing')):
            first = True
            for m in ('nox', 'nox-tuned-lenient' if part == 'e3' else 'nox-tuned-strict', 'vela-0.8b', 'vela-4b', 'vela-9b'):
                r = VR.get(f'{part}/{m}')
                if not r: continue
                ms_ = NUM.get(f'vf:{K[part]}:{m}:small_ms', NUM.get(f'vf:{K[part]}:nox:small_ms', '--'))
                f.write(f"{nm if first else ''} & {lab[m]} & {r['small_acc']:.1f} & {r['cascade_acc']:.1f} & "
                        f"{r['d_big']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] & {r['esc']:.0f} & {ms_} & {r['speedup']:.2f}$\\times$ \\\\\n")
                first = False
            f.write('\\midrule\n' if part == 'e3' else '')
        f.write('\\bottomrule\n\\end{tabular}\n')

with open(OUT / 'numbers.tex', 'w') as f:
    f.write('% generated by clef/eval/ft_paper.py -- do not edit\n')
    for key, v in sorted(NUM.items()):
        f.write(f'\\expandafter\\def\\csname num:{key}\\endcsname{{{v}}}\n')
json.dump(J, open(OUT / 'numbers.json', 'w'), indent=1, default=str)
print(f'{len(NUM)} numbers; live parts: {list(LV)}')
for part, r in LV.items():
    print(part, {x: (round(v, 2) if isinstance(v, float) else v) for x, v in r.items()})
