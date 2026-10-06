"""Run one model on one experiment of the frozen dataset; results go beside the data.

    python3 run.py MODEL EXP [EXP ...]      # MODEL in tasks.MODELS; EXP in e1 e2 e3 e4 e5
    DATA=~/data/decision-eval/v1 (default)

Writes $DATA/results/<exp-part>/<model>.json. An existing result file is never overwritten
(delete it to rerun). SystemOne models run single-stream so latency is comparable; the
generative model runs 8-way concurrent as in the original runs (6-way with reasoning).
"""
import json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as T

DATA = Path(os.path.expanduser(os.environ.get('DATA', '~/data/decision-eval/v1')))
model, exps = sys.argv[1], sys.argv[2:]
assert model in T.MODELS, f'unknown model {model}; one of {T.MODELS}'
CHATTY = model in T.CHAT
load = lambda name: json.load(open(DATA / f'{name}.json'))


def timed(fn):
    t = time.time(); out = fn(); return out, (time.time() - t) * 1000


def so(state, qs):
    (ans, mid, usage), ms = timed(lambda: T.systemone(model, state, qs))
    return dict(answers=ans, served=mid, input_tokens=usage.get('input_tokens'), ms=ms)


def run(part, rows, fn, keep, threads=None):
    out = DATA / 'results' / part / f'{model}.json'
    if out.exists():
        print(f'skip {part}/{model} (exists)'); return
    out.parent.mkdir(parents=True, exist_ok=True)
    n = threads or (1 if not CHATTY else 6 if model == 'qwen-think' else 8)

    def one(r):
        for i in range(3):
            try:
                return {**{k: r[k] for k in keep if k in r}, **fn(r)}
            except Exception:
                if i == 2: traceback.print_exc(); return {**{k: r[k] for k in keep if k in r}, 'error': True}
                time.sleep(5)
    t = time.time(); res = list(ThreadPoolExecutor(n).map(one, rows))
    errs = sum('error' in r for r in res)
    json.dump(res, open(out, 'w'), indent=1)
    print(f'{part}/{model}: {len(res)} rows, {errs} errors, {time.time() - t:.0f}s')


def chat_choice(state, q, maxtok=60):
    (t), ms = timed(lambda: T.chat(T.choice_chat_prompt(state, q), False, maxtok))
    return dict(p=T.choice_chat_parse(t, list(q['kind']['criteria'])), ms=ms)


for e in exps:
    if e == 'e1':
        if CHATTY: print('e1: SystemOne models only'); continue
        run('e1_prose', load('e1_cards_prose'), lambda r: so(r['state'], T.Q_E1_PROSE), ('session', 'id', 'verdict', 'note'))
        for arm in ('B', 'C'):
            run(f'e1_raw{arm}', load(f'e1_cards_raw{arm}'), lambda r: so(r['state'], T.Q_E1_RAW), ('session', 'id', 'verdict', 'note'))
    elif e == 'e2':
        keep = ('session', 'id', 'token', 'gold', 'gold_in_opts')
        if CHATTY:
            think = T.CHAT[model]
            def f(r):
                t, ms = timed(lambda: T.chat(T.e2_chat_prompt(r), think, 4000 if think else 200))
                return dict(ans=T.e2_parse(t), ms=ms, raw=t[-300:])
        else:
            def f(r):
                x = so(r['state'], T.e2_questions(r)); a = x.pop('answers')['pick']
                return dict(ans=dict(choice=a['choice'], confidence=a.get('confidence', 0), probabilities=a['probabilities']), **x)
        run('e2', load('e2_pick'), f, keep)
    elif e == 'e3':
        if model == 'qwen-think': continue
        keep = ('session', 'token', 'label', 'n')
        if CHATTY:
            f = lambda r: chat_choice(r['state'], T.Q_TRIAGE)
        else:
            def f(r):
                x = so(r['state'], T.Q_TRIAGE); return dict(p=x.pop('answers')['kind']['probabilities'], **x)
        run('e3', load('e3_triage'), f, keep)
    elif e == 'e4':
        if model == 'qwen-think': continue
        for part, src, q in (('e4_ent', 'e4_ent', T.Q_ENT), ('e4_facts', 'e4_facts', T.Q_FACT), ('e4_ctrl', 'e4_ctrl', T.Q_FACT)):
            rows = load(src); keep = tuple(k for k in rows[0] if k != 'state') + ('member', 'status', 'votes', 'members')
            if CHATTY:
                f = lambda r, q=q: chat_choice(r['state'], q)
            else:
                def f(r, q=q):
                    x = so(r['state'], q); return dict(p=x.pop('answers')['kind']['probabilities'], **x)
            run(part, rows, f, keep)
    elif e == 'e5':
        if model == 'qwen-think': continue
        def f(r):
            res = {}
            for side in ('a', 'b'):
                st = r[side]['state']
                if CHATTY:
                    t, ms = timed(lambda: T.chat(T.npc_chat_prompt(st), False, 400))
                    res[side] = dict(T.npc_chat_parse(t), ms=ms)
                else:
                    x = so(st, T.Q_NPC); a = x.pop('answers')
                    res[side] = dict(outcome=a['outcome']['probabilities'], target=a['target']['probabilities'], **x)
            return res
        run('e5', load('e5_npc'), f, ('id', 'metric', 'sign'), threads=1)
    else:
        sys.exit(f'unknown experiment {e}')
