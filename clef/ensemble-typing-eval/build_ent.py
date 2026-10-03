# Per-entity type test. Gold = GM's primary type in .type_merge_decisions.json (merged groups);
# kept_separate groups are tested for NOT collapsing. Evidence = the raw extracted facts
# (merged.json) of every member, with qwen's original type labels hidden.
import json,re,random,collections
E='/home/kostadis/out-of-the-abyss/out-of-the-abyss/docs/ensemble'
from facts_index import index
a=index('per_chapter'); b=index('per_chapter_rerun')
facts={k:a.get(k) or b.get(k) for k in set(a)|set(b)}
G=json.load(open(f'{E}/.type_merge_decisions.json'))['groups']
random.seed(7); rows=[]; missing=0
def member_facts(m):
    t,s=m[:-3].split('_',1); return t,facts.get((t,s),[])
for g in G:
    for r in g.get('resolution',[]):
        mem=[member_facts(m)+(m,) for m in r['members']]
        if any(not fs for _,fs,_ in mem): missing+=1
        if all(not fs for _,fs,_ in mem): continue
        votes=collections.Counter(); pool=[]
        for t,fs,m in mem:
            votes[t]+=len(fs); pool+= [(m,f) for f in fs]
        subj=collections.Counter(f['subject'] for _,f in pool).most_common(1)[0][0] if pool else g['key']
        def state_of(fs):
            fs=fs[:]; random.shuffle(fs)
            return f"Entity: {subj}\n\nFacts extracted about it from a D&D campaign's session records:\n"+'\n'.join('- '+f['fact'] for f in fs[:14])
        base=dict(key=g['key'],status=r['status'],members=r['members'],votes=dict(votes))
        if r['status']=='merged':
            rows.append({**base,'gold':r['primary'].split('_')[0],'state':state_of([f for _,f in pool])})
        else:  # one item per member; each should keep its own type
            for t,fs,m in mem:
                if fs: rows.append({**base,'gold':t,'member':m,'state':state_of(fs).replace(f"Entity: {subj}",f"Entity: {fs[0]['subject']}")})
json.dump(rows,open('ent.json','w'),indent=1)
print(len(rows),'items;',sum(r['status']=='merged' for r in rows),'merged groups;',missing,'groups with a member lacking facts')
mv=[r for r in rows if r['status']=='merged']
print('majority-vote baseline on merged groups:',sum(max(r['votes'],key=r['votes'].get)==r['gold'] for r in mv),'/',len(mv))
print(rows[3]['state'][:800])
