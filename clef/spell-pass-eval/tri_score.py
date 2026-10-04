import json,os
import sys
for w in (sys.argv[1:] or ('clef-flash','clef','qwen')):
    if not os.path.exists(f'tri_{w}.json'): continue
    o=json.load(open(f'tri_{w}.json')); N=len(o); names=[r for r in o if r['label']=='name']
    ms=sorted(r['ms'] for r in o)[N//2]
    print(f"== {w}  (median {ms:.0f} ms; {len(names)} real names, {N} tokens)")
    for sname,f in [('P(campaign)',lambda p:p['campaign']),('not filler/realworld',lambda p:1-p['filler']-p['realworld'])]:
        for th in ([0.5] if w=='qwen' else [0.02,0.05,0.1,0.2,0.3,0.5]):
            sent=[r for r in o if f(r['p'])>=th]; miss=[r['token'] for r in names if f(r['p'])<th]
            ch=sum(r['label']=='chatter' for r in o if f(r['p'])<th); lv=sum(r['label']=='leave' for r in o if f(r['p'])<th)
            print(f"  {sname:22s} >= {th:<4}: sends {len(sent):3d}/{N} ({len(sent)/N:4.0%}), cleared {lv} left-alone + {ch} chatter, MISSES {len(miss)} names {miss[:8]}")
    if w!='qwen':
        top=lambda p:max(p,key=p.get)
        ch=[r for r in o if r['label']=='chatter']; print(f"  chatter recognised as realworld: {sum(top(r['p'])=='realworld' for r in ch)}/{len(ch)}")
    else:
        ch=[r for r in o if r['label']=='chatter']; print(f"  chatter recognised as realworld: {sum(r['p']['realworld']==1 for r in ch)}/{len(ch)}")
