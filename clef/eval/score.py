"""Score every result in $DATA/results the way the paper reports it; prints one block per table.

    python3 score.py [--json OUT]     # DATA=~/data/decision-eval/v1 (default)
"""
import collections, json, os, re, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as T

DATA = Path(os.path.expanduser(os.environ.get('DATA', '~/data/decision-eval/v1')))
R = DATA / 'results'
ORDER = ['clef', 'clef-flash', 'lux', 'nox', 'kai', 'jev', 'qwen', 'qwen-think']
OUT = {}
load = lambda p: json.load(open(p))


def results(part):
    d = R / part
    return {m: load(d / f'{m}.json') for m in ORDER if (d / f'{m}.json').exists()}


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float('nan')


def auc(pos, neg):
    if not pos or not neg: return float('nan')
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def emit(table, model, **kv):
    OUT.setdefault(table, {})[model] = kv
    print(f'  {model:11s} ' + '  '.join(f'{k}={v}' for k, v in kv.items()))


# ---- E1 ------------------------------------------------------------------------------------
print('== E1: AUC of P(accept) for approve vs not')
rawB = {(r['session'], r['id']): r['state'] for r in load(DATA / 'e1_cards_rawB.json')}
def has_fix(k):
    s = rawB.get(k, '')
    tok = re.search(r'Transcript token: (.*)', s); fix = re.search(r'Proposed replacement: (.*)', s)
    tok = tok.group(1).strip() if tok else ''; fix = fix.group(1).strip() if fix else ''
    return bool(fix) and fix.lower() not in ('', tok.lower(), '(none)', 'none')
cards = load(DATA / 'e1_cards_prose.json')
fx = [has_fix((r['session'], r['id'])) for r in cards]
ok = [r['verdict'] == 'approve' for r in cards]
print(f'  cards {len(cards)}, not approved {ok.count(False)}; with fix {sum(fx)} ({sum(1 for f, o in zip(fx, ok) if f and not o)} not approved), '
      f'no fix {len(fx) - sum(fx)} ({sum(1 for f, o in zip(fx, ok) if not f and not o)} not approved); '
      f'baseline AUC has-fix {auc([float(f) for f, o in zip(fx, ok) if o], [float(f) for f, o in zip(fx, ok) if not o]):.3f}')
OUT['e1_baseline'] = dict(n=len(cards), not_approved=ok.count(False), with_fix=sum(fx),
                          auc=auc([float(f) for f, o in zip(fx, ok) if o], [float(f) for f, o in zip(fx, ok) if not o]))
for part in ('e1_prose', 'e1_rawB', 'e1_rawC'):
    print(f' -- {part}')
    for m, rs in results(part).items():
        row = {}
        for sub, sel in (('all', lambda f: True), ('fix', lambda f: f), ('nofix', lambda f: not f)):
            p = [r['answers']['accept']['noul'] for r in rs if 'error' not in r and sel(has_fix((r['session'], r['id']))) and r['verdict'] == 'approve']
            n = [r['answers']['accept']['noul'] for r in rs if 'error' not in r and sel(has_fix((r['session'], r['id']))) and r['verdict'] != 'approve']
            row[sub] = round(auc(p, n), 3)
        emit(part, m, **row, median_ms=round(med([r.get('ms') for r in rs])))

# ---- E2 ------------------------------------------------------------------------------------
print('== E2: canonical spelling')
E2 = results('e2')
def e2rows(rs):
    for r in rs:
        a = r.get('ans') or {}
        r['pred'] = a.get('choice'); r['conf'] = a.get('confidence') or 0
        r['target'] = r['gold'] if r['gold_in_opts'] else 'OTHER'
        r['ok'] = r['pred'] == r['target']
        r['wrongname'] = (not r['ok']) and r['pred'] not in ('OTHER', 'LEAVE', None)
    return rs
for m, rs in E2.items():
    e2rows(rs); n = len(rs); ans = [r for r in rs if r['pred'] is not None]
    s = sorted(rs, key=lambda r: -r['conf']); h = s[:n // 2]
    emit('e2', m, n=n, acc=f"{sum(r['ok'] for r in rs) / n:.1%}", acc_answered=f"{sum(r['ok'] for r in ans) / max(1, len(ans)):.1%} of {len(ans)}",
         wrong_names=sum(r['wrongname'] for r in rs), unanswered=n - len(ans),
         unanswered_gold_missing=sum(1 for r in rs if r['pred'] is None and not r['gold_in_opts']),
         conf_top_half_acc=f"{sum(r['ok'] for r in h) / len(h):.1%}", median_ms=round(med([r['ms'] for r in (ans or rs)])))
print(f"  candidate recall: gold in list for {sum(r['gold_in_opts'] and r['gold'] != 'LEAVE' for r in next(iter(E2.values())))} "
      f"of {sum(r['gold'] != 'LEAVE' for r in next(iter(E2.values())))} names" if E2 else '')
for a in [m for m in E2 if m not in ('qwen', 'qwen-think')]:
    if 'qwen-think' not in E2: break
    ag = [(x, y) for x, y in zip(E2[a], E2['qwen-think']) if x['pred'] == y['pred']]
    emit('e2_agree', f'{a}+think', agree=len(ag), acc=f"{sum(x['ok'] for x, _ in ag) / max(1, len(ag)):.1%}", wrong_names=sum(x['wrongname'] for x, _ in ag))

# ---- E3 ------------------------------------------------------------------------------------
print('== E3: triage (forwarded share at k missed names)')
SCORES = {'P(campaign)': lambda p: p['campaign'], 'not filler/realworld': lambda p: 1 - p['filler'] - p['realworld']}
def fwd_at(rs, f, k):
    names = sorted(f(r['p']) for r in rs if r['label'] == 'name')
    th = names[k]
    return sum(f(r['p']) >= th for r in rs) / len(rs)
for m, rs in results('e3').items():
    rs = [r for r in rs if 'error' not in r]
    ch = [r for r in rs if r['label'] == 'chatter']
    rec = sum(max(r['p'], key=r['p'].get) == 'realworld' and r['p']['realworld'] > 0 for r in ch)
    if m.startswith('qwen'):
        fl = [r for r in rs if r['p']['campaign'] == 1]; miss = sum(1 for r in rs if r['label'] == 'name' and r['p']['campaign'] != 1)
        emit('e3', m, forwarded=f'{len(fl) / len(rs):.0%}', missed=miss, chatter=f'{rec}/{len(ch)}', median_ms=round(med([r['ms'] for r in rs])))
        continue
    z = [fwd_at(rs, f, 0) for f in SCORES.values()]; two = [fwd_at(rs, f, 2) for f in SCORES.values()]
    # held-out: threshold = half the lowest name score on one half of the sessions, applied to the other
    sess = sorted({r['session'] for r in rs}); halves = [set(sess[0::2]), set(sess[1::2])]; ho = []
    for tr, te in ((halves[0], halves[1]), (halves[1], halves[0])):
        th = 0.5 * min(r['p']['campaign'] for r in rs if r['session'] in tr and r['label'] == 'name')
        tes = [r for r in rs if r['session'] in te]
        ho.append(f"{sum(r['p']['campaign'] >= th for r in tes) / len(tes):.0%}/{sum(1 for r in tes if r['label'] == 'name' and r['p']['campaign'] < th)}miss")
    lo = lambda xs: f'{min(xs):.0%}' if round(min(xs), 2) == round(max(xs), 2) else f'{min(xs):.0%}-{max(xs):.0%}'
    emit('e3', m, fwd_0miss=lo(z), fwd_le2miss=lo(two), auc=round(auc([r['p']['campaign'] for r in rs if r['label'] == 'name'],
         [r['p']['campaign'] for r in rs if r['label'] != 'name']), 3), heldout=' '.join(ho), chatter=f'{rec}/{len(ch)}',
         median_ms=round(med([r['ms'] for r in rs])))

# ---- E4 ------------------------------------------------------------------------------------
print('== E4: entity typing')
ENT = {'npc', 'monster', 'faction', 'location', 'object'}
top = lambda d: max(d, key=d.get)
def splits(rows, key):
    by = collections.defaultdict(set)
    for r in rows:
        if key(r) in ENT: by[r['key']].add(key(r))
    return sum(len(v) > 1 for v in by.values())
for m, rs in results('e4_ent').items():
    ent = [r for r in rs if r['status'] == 'merged' and 'error' not in r]
    alone = sum(top(r['p']) == r['gold'] for r in ent); vote = sum(top(r['votes']) == r['gold'] for r in ent)
    hyb = sum((top(r['votes']) if max(r['votes'].values()) / sum(r['votes'].values()) >= .6 else top(r['p'])) == r['gold'] for r in ent)
    flags = [r for r in ent if top(r['p']) != top(r['votes'])]
    emit('e4_ent', m, n=len(ent), vote=vote, model_alone=alone, vote60_else_model=hyb, flags=len(flags),
         caught=f"{sum(top(r['votes']) != r['gold'] for r in flags)}/{sum(top(r['votes']) != r['gold'] for r in ent)}",
         median_ms=round(med([r['ms'] for r in rs])))
facts, ctrl = results('e4_facts'), results('e4_ctrl')
base = load(DATA / 'e4_facts.json'); cbase = load(DATA / 'e4_ctrl.json')
print(f"  extraction as is: split {splits(base, lambda r: r['orig'])} of {len({r['key'] for r in base})}; control "
      f"{splits(cbase, lambda r: r['orig'])} of {len({r['key'] for r in cbase})} ({len(cbase)} facts); "
      f"mistyped facts {sum(r['orig'] != r['gold'] for r in base)} of {len(base)}")
def entvote(rows, key):
    by = collections.defaultdict(collections.Counter); gold = {}
    for r in rows:
        if key(r) in ENT: by[r['key']][key(r)] += 1
        gold[r['key']] = r['gold']
    return sum(1 for k, c in by.items() if c and c.most_common(1)[0][0] == gold[k])
print(f"  vote over extraction's per-fact types on this fact set: {entvote(base, lambda r: r['orig'])}/{len({r['key'] for r in base})}")
for m in facts:
    f = [dict(r, pred=top(r['p'])) for r in facts[m] if 'error' not in r]
    c = [dict(r, pred=top(r['p'])) for r in ctrl.get(m, []) if 'error' not in r]
    mv = lambda rows: sum(r['pred'] in ('event', 'thread', 'date') for r in rows)
    pooled = [dict(r, pred=r['orig']) for r in base] + f
    emit('e4_fact', m, acc=f"{sum(r['pred'] == r['gold'] for r in f) / len(f):.1%}", split_repair=splits(f, lambda r: r['pred']),
         new_split_ctrl=splits(c, lambda r: r['pred']) if c else None, moved=f'{mv(f)}+{mv(c)}',
         vote_over_model=entvote(f, lambda r: r['pred']), vote_pooled=entvote(pooled, lambda r: r['pred']),
         median_ms=round(med([r['ms'] for r in f])))

# ---- E5 ------------------------------------------------------------------------------------
print('== E5: NPC pairs moving the expected way (delta > 0.02)')
pairs = load(DATA / 'e5_npc.json')
def wins(get):
    w = 0
    for P in pairs:
        kind, key = P['metric']; a, b = get(P, 'a')[kind].get(key, 0), get(P, 'b')[kind].get(key, 0)
        w += (b - a) * P['sign'] > 0.02
    return w
emit('e5', 'flexai', moved=f"{wins(lambda P, s: P[s]['flexai'])}/{len(pairs)}")
for m, rs in results('e5').items():
    by = {r['id']: r for r in rs}
    emit('e5', m, moved=f"{wins(lambda P, s: by[P['id']][s])}/{len(pairs)}",
         goblin_flee=' -> '.join(f"{by[i][s]['outcome']['flee']:.2f}" for i, s in (('goblin_morale', 'a'), ('goblin_morale', 'b'), ('cornered_goblin', 'b'))),
         median_ms=round(med([by[P['id']][s].get('ms') for P in pairs for s in 'ab'])))

# ---- E7: latency summary ---------------------------------------------------------------------
print('== E7: median latency by experiment (ms)')
for m in ORDER:
    row = {p: round(med([r.get('ms') for r in load(R / p / f'{m}.json')])) for p in
           ('e1_prose', 'e2', 'e3', 'e4_ent', 'e4_facts') if (R / p / f'{m}.json').exists()}
    if row: emit('e7', m, **row)

if '--json' in sys.argv:
    json.dump(OUT, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
