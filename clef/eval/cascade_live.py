"""Run the Nox -> qwen cascade live, against the real endpoints, on the rows the gate file covers.

    python3 cascade_live.py MODE PART [PART ...] [--conc N] [--limit K]
        MODE  small | big | cascade
        PART  e2 e3 e4_ent e4_facts
    DATA=~/data/decision-eval/v1 (default); gates from $DATA/cascade/gates-nox-qwen.json (cascade.py --gates)

small   = Nox alone (spark2:8005), big = qwen3.8-flash-next alone (spark1:8001, reasoning off),
cascade = a real two-stage router: ask Nox, and if its gate confidence is below the row's pre-registered
          threshold, ask qwen the same question and answer with qwen.
Questions, prompts and parsers are tasks.py's, called exactly as run.py calls them, so a live answer is
directly comparable with the frozen result for the same row. Writes
$DATA/cascade/live/<part>/<mode>-c<N>.json (never overwrites) with per-row answers and timings plus the
run's wall clock, so both latency (c=1) and throughput (c>1) can be read off.
"""
import argparse, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as T

DATA = Path(os.path.expanduser(os.environ.get('DATA', '~/data/decision-eval/v1')))
ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=('small', 'big', 'cascade')); ap.add_argument('parts', nargs='+')
ap.add_argument('--conc', type=int, default=1); ap.add_argument('--limit', type=int)
ap.add_argument('--tag', default='', help='suffix for the output file, naming the serving config (e.g. -mtp0)')
ap.add_argument('--gates', default=str(DATA / 'cascade' / 'gates-nox-qwen.json'))
a = ap.parse_args()
G = json.load(open(a.gates)); SMALL = G['small']
assert G['big'] == 'qwen', 'live big model is qwen3.8-flash-next, reasoning off'
top = lambda d: max(d, key=d.get)
now = lambda: time.perf_counter() * 1000

# part -> (frozen input, SystemOne questions(row), answer key, chat prompt(row), chat parse(text) -> pred,
#          small answer -> (pred, gate confidence), chat max_tokens)
def _e2_small(ans):
    a = ans['pick']; return a['choice'], max(a['probabilities'].values())
def _e3_small(ans):
    p = ans['kind']['probabilities']; return top(p) == 'campaign', max(p['campaign'], 1 - p['campaign'])
def _e4_small(ans):
    p = ans['kind']['probabilities']; return top(p), max(p.values())
def _choice(q, as_bool=False):
    keys = list(q['kind']['criteria'])
    def parse(t):
        k = top(T.choice_chat_parse(t, keys)); return k == 'campaign' if as_bool else k
    return lambda r: T.choice_chat_prompt(r['state'], q), parse

PARTS = {
    'e2':       ('e2_pick',   T.e2_questions,       _e2_small, lambda r: T.e2_chat_prompt(r), lambda t: T.e2_parse(t)['choice'], 200),
    'e3':       ('e3_triage', lambda r: T.Q_TRIAGE, _e3_small, *_choice(T.Q_TRIAGE, True), 60),
    'e4_ent':   ('e4_ent',    lambda r: T.Q_ENT,    _e4_small, *_choice(T.Q_ENT), 60),
    'e4_facts': ('e4_facts',  lambda r: T.Q_FACT,   _e4_small, *_choice(T.Q_FACT), 60),
}


def small(part, r):
    _, qs, ext, *_ = PARTS[part]
    t = now(); ans, mid, usage = T.systemone(SMALL, r['state'], qs(r)); ms = now() - t
    pred, conf = ext(ans)
    return dict(pred=pred, conf=conf, ms=ms, in_tok=usage.get('input_tokens'))


def big(part, r):
    *_, prompt, parse, maxtok = PARTS[part]
    t = now(); txt = T.chat(prompt(r), False, maxtok); ms = now() - t
    return dict(pred=parse(txt), ms=ms)


def cascade(part, r, gate):
    t = now(); s = small(part, r)
    out = dict(small_pred=s['pred'], conf=s['conf'], small_ms=s['ms'], t=gate['t'], esc=s['conf'] < gate['t'])
    if out['esc']:
        b = big(part, r); out.update(big_pred=b['pred'], big_ms=b['ms'], pred=b['pred'])
    else:
        out['pred'] = s['pred']
    out['ms'] = now() - t
    return out


for part in a.parts:
    out = DATA / 'cascade' / 'live' / part / f'{a.mode}-c{a.conc}{a.tag}.json'
    if out.exists() and not a.limit:
        print(f'skip {out} (exists)'); continue
    rows = json.load(open(DATA / f'{PARTS[part][0]}.json'))
    gates = G['gates'][part]
    idx = sorted(int(i) for i in gates)[:a.limit]

    def one(i):
        r = rows[i]
        for k in range(3):
            try:
                t0 = time.time()
                res = (small(part, r) if a.mode == 'small' else big(part, r) if a.mode == 'big'
                       else cascade(part, r, gates[str(i)]))
                return dict(i=i, start=t0, **res)
            except Exception:
                if k == 2: traceback.print_exc(); return dict(i=i, error=True)
                time.sleep(5)
    t0 = time.time(); res = list(ThreadPoolExecutor(a.conc).map(one, idx)); wall = time.time() - t0
    doc = dict(mode=a.mode, part=part, conc=a.conc, small=SMALL, big=T.CHAT_MODEL, wall_s=wall, n=len(res),
               errors=sum('error' in r for r in res), started=t0, gates_file=a.gates, tag=a.tag, rows=res)
    if a.limit:
        print(json.dumps({k: v for k, v in doc.items() if k != 'rows'}), json.dumps(res[:2])[:400]); continue
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(doc, open(out, 'w'), indent=1)
    print(f'{part} {a.mode} c{a.conc}: {len(res)} rows, {doc["errors"]} errors, {wall:.0f}s', flush=True)
