import json,collections,os
ENT={'npc','monster','faction','location','object'}
def bucket(t,g): return 'match' if t==g else 'evt' if t in('event','thread','date') else 'other'
def splits(rows,key):
    by=collections.defaultdict(set)
    for r in rows:
        t=key(r)
        if t in ENT: by[r['key']].add(t)
    return sum(len(v)>1 for v in by.values())
base=json.load(open('facts.json'))
print(f"{'source':12s} match  evt/thr/date  other-entity | fixes of qwen's 266 | breaks of qwen's 1425 | entities split | median ms")
print(f"{'extraction':12s} {sum(r['orig']==r['gold'] for r in base):5d}  {0:12d}  {sum(bucket(r['orig'],r['gold'])=='other' for r in base):12d} | {'—':>18s} | {'—':>21s} | {splits(base,lambda r:r['orig']):14d} |")
for w in ('clef-flash','clef','qwen'):
    p=f'facts_{w}.json'
    if not os.path.exists(p): continue
    o=json.load(open(p)); c=collections.Counter(bucket(r['pred'],r['gold']) for r in o)
    fx=sum(r['pred']==r['gold'] for r in o if r['orig']!=r['gold']); br=collections.Counter(bucket(r['pred'],r['gold']) for r in o if r['orig']==r['gold'])
    ms=sorted(r['ms'] for r in o)[len(o)//2]
    print(f"{w:12s} {c['match']:5d}  {c['evt']:12d}  {c['other']:12d} | {fx:18d} | {br['other']:>6d} other, {br['evt']:4d} evt | {splits(o,lambda r:r['pred']):14d} | {ms:.0f}")
    # does high confidence select the reliable ones?
    if w!='qwen':
        s=sorted(o,key=lambda r:-max(r['p'].values()))
        for f in (.5,.8): k=int(len(s)*f); print(f"     most-confident {int(f*100)}%: other-entity errors {sum(bucket(r['pred'],r['gold'])=='other' for r in s[:k])}/{k}")
