import json,re,sys
def auc(pos,neg):
    if not pos or not neg: return float('nan')
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
for m in sys.argv[1:]:
    o=json.load(open(f'out_{m}.json'))
    ischat=lambda r: r['verdict']=='chatter' or bool(re.search(r'chatter|table talk|real.?world|^\s*table\b',r['note'],re.I))
    acc=[r['resp']['answers']['accept']['noul'] for r in o]
    ch=[r['resp']['answers']['chatter']['noul'] for r in o]
    appr=[a for a,r in zip(acc,o) if r['verdict']=='approve']; nap=[a for a,r in zip(acc,o) if r['verdict']!='approve']
    cp=[c for c,r in zip(ch,o) if ischat(r)]; cn=[c for c,r in zip(ch,o) if not ischat(r)]
    ms=sorted(r['ms'] for r in o)
    print(f"== {m}: n={len(o)} median {ms[len(ms)//2]:.0f}ms")
    print(f" accept: AUC approve-vs-not {auc(appr,nap):.3f}  (approve n={len(appr)}, not n={len(nap)})")
    print(f" chatter: AUC {auc(cp,cn):.3f} (chatter n={len(cp)})")
    # triage: lowest-accept 20% — what fraction of non-approves caught?
    srt=sorted(zip(acc,o),key=lambda x:x[0])
    for frac in (.1,.2,.3):
        k=int(len(srt)*frac); caught=sum(r['verdict']!='approve' for _,r in srt[:k])
        print(f"  bottom {int(frac*100)}% by accept ({k} cards) holds {caught}/{len(nap)} non-approves")
    for th in (.5,.8,.9):
        fl=[r for c,r in zip(ch,o) if c>=th]
        print(f"  chatter>={th}: {len(fl)} flagged, {sum(ischat(r) for r in fl)} true chatter; {sum(r['verdict']=='approve' for r in fl)} were approved corrections")
