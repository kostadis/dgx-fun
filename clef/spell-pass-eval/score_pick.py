import json,sys,os
def load(w):
    o=json.load(open(f'pick_{w}.json'))
    for r in o:
        a=r['ans']; r['pred']=a.get('choice'); r['conf']=a.get('confidence',a.get('conf',0)) or 0
        r['target']=r['gold'] if r['gold_in_opts'] else 'OTHER'
        r['ok']=r['pred']==r['target']
        r['wrongname']=(not r['ok']) and r['pred'] not in ('OTHER','LEAVE',None)
    return o
def report(name,o):
    n=len(o); ok=sum(r['ok'] for r in o); wn=sum(r['wrongname'] for r in o)
    io=[r for r in o if r['gold_in_opts'] and r['gold']!='LEAVE']; lv=[r for r in o if r['gold']=='LEAVE']; ot=[r for r in o if not r['gold_in_opts']]
    ms=sorted(r['ms'] for r in o)[n//2]
    print(f"{name:12s} acc {ok/n:5.1%}  | right name in list {sum(r['ok'] for r in io)}/{len(io)}  | 'not in list' {sum(r['ok'] for r in ot)}/{len(ot)}  | chatter {sum(r['ok'] for r in lv)}/{len(lv)}  | WRONG NAME {wn} | unparsed {sum(r['pred'] is None for r in o)} | median {ms:.0f}ms")
    s=sorted(o,key=lambda r:-r['conf'])
    for f in (.5,.75,.9):
        k=int(n*f); print(f"     most-confident {int(f*100)}%: acc {sum(r['ok'] for r in s[:k])/k:5.1%}, wrong names {sum(r['wrongname'] for r in s[:k])}")
W=[w for w in ('clef-flash','clef','qwen','qwen-think') if os.path.exists(f'pick_{w}.json')]
D={w:load(w) for w in W}
for w in W: report(w,D[w])
if 'clef' in D and 'qwen' in D:
    for q in [w for w in ('qwen','qwen-think') if w in D]:
        for t in (.6,.8,.9):
            comb=[]
            for c,qq in zip(D['clef'],D[q]):
                comb.append(c if c['conf']>=t else qq)
            n=len(comb); print(f"clef>={t} else {q}: acc {sum(r['ok'] for r in comb)/n:5.1%}, wrong names {sum(r['wrongname'] for r in comb)}")
        agree=[(c,qq) for c,qq in zip(D['clef'],D[q]) if c['pred']==qq['pred']]
        print(f"clef & {q} agree on {len(agree)}/{len(D[q])}: acc when agree {sum(c['ok'] for c,_ in agree)/max(1,len(agree)):5.1%}, wrong names {sum(c['wrongname'] for c,_ in agree)}")
