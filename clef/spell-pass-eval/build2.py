# Arm B: rebuild each GM-ruled card from raw sources only (no Claude-written card text).
import json, glob, re, os, subprocess
from concurrent.futures import ThreadPoolExecutor

C = '/home/kostadis/out-of-the-abyss/out-of-the-abyss'
SIB = os.path.expanduser('~/.claude/skills/vtt-spell-pass/sibling_context.py')
canon_txt = {'party file': open(f'{C}/config/party.yaml').read(),
             'entity registry': open(f'{C}/docs/entity_registry.yaml').read()}
doss = [os.path.basename(p)[:-3].replace('_', ' ').lower() for p in glob.glob(f'{C}/docs/distill/npcs/*.md')]


def cues(path):
    txt = open(path, errors='ignore').read()
    out = []
    for blk in re.split(r'\n\s*\n', txt):
        ls = [l for l in blk.strip().split('\n')
              if l and '-->' not in l and not l.strip().isdigit() and l.strip() != 'WEBVTT']
        if ls:
            out.append(' '.join(ls))
    return out


cache = {}
def get(p):
    if p not in cache:
        cache[p] = cues(p)
    return cache[p]


def first(x):
    x = x[0] if isinstance(x, list) and x else x
    return x if isinstance(x, str) else ''


rows, jobs = [], []
for f in sorted(glob.glob(f'{C}/summaries/*/spell_review/decisions.json')):
    sd = os.path.dirname(os.path.dirname(f))
    dec = json.load(open(f))
    mp = json.load(open(f.replace('decisions', 'review_map')))
    tapes = [p for p in glob.glob(f'{sd}/*.vtt') if '.cleaned' not in p and '.speakers' not in p and 'RAW' not in p]
    for k, v in dec['decisions'].items():
        m = mp[k]
        tok, fix = first(m.get('token')).strip(), first(m.get('canonical')).strip()
        prim, ctx, n = None, '(token not found verbatim on the raw tape)', 0
        if tok:
            for p in sorted(tapes, key=lambda p: (os.path.basename(p) != 'transcript.vtt', p)):
                cs = get(p)
                hits = [i for i, c in enumerate(cs) if tok.lower() in c.lower()]
                if hits:
                    prim, i, n = p, hits[0], len(hits)
                    ctx = '\n'.join(cs[max(0, i - 2):i + 3])
                    break
        sib = [p for p in tapes if p != prim]
        if sib and prim:
            jobs.append((len(rows), sib[0], ctx[-600:]))
        where = [nm for nm, t in canon_txt.items() if fix and fix in t] + (['NPC dossier'] if fix.lower() in doss else [])
        rows.append(dict(session=os.path.basename(sd), id=k, verdict=v, note=dec['notes'].get(k, ''),
                         tok=tok, fix=fix, n=n, ctx=ctx, where=where))


def lookup(j):
    i, p, c = j
    r = subprocess.run(['python3', SIB, '--sibling', p, '--context', c, '--top', '1'], capture_output=True, text=True)
    return i, (r.stdout.strip()[:900] or '(no output)')


res = dict(ThreadPoolExecutor(14).map(lookup, jobs))
for i, r in enumerate(rows):
    sibtxt = res.get(i, '(no second transcription for this session)')
    r['state'] = (f"Transcript token: {r['tok']}\nProposed replacement: {r['fix']}\nOccurrences on tape: {r['n']}\n\n"
                  f"Raw transcript around first occurrence:\n{r['ctx']}\n\n"
                  f"Second, independent transcription at the same span:\n{sibtxt}\n\n"
                  f"Proposed replacement found in canon sources: {', '.join(r['where']) or 'none'}")
    for x in ('tok', 'fix', 'n', 'ctx', 'where'):
        r.pop(x)
json.dump(rows, open('cards2.json', 'w'), indent=1)
print(len(rows), 'cards;', len(jobs), 'sibling lookups')
print(rows[300]['state'])
