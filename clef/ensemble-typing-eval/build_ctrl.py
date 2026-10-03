# Control set: entities qwen typed CONSISTENTLY (one dossier, never in a type-merge group).
# The merged-group set was selected on qwen's mistakes, so any retyper looks good there;
# this measures how many NEW splits a retyper would create on entities that were fine.
import json,collections,random,glob,os
from facts_index import index,C
a=index('per_chapter'); G=json.load(open(f'{C}/docs/ensemble/.type_merge_decisions.json'))['groups']
inG={m[:-3] for g in G for r in g['resolution'] for m in r['members']}
subj_types=collections.defaultdict(set)
for (t,s) in a:
    if t in('npc','monster','faction','location','object'): subj_types[s].add(t)
dossiers={os.path.basename(p)[:-3] for p in glob.glob(f'{C}/docs/ensemble/state_dossiers/*.md')}
cands=[(t,s) for (t,s),fs in a.items() if f'{t}_{s}' in dossiers and f'{t}_{s}' not in inG and len(subj_types[s])==1 and 8<=len(fs)<=40]
random.seed(3); pick=random.sample(cands,min(70,len(cands)))
rows=[]
for t,s in pick:
    for f in a[(t,s)]:
        rows.append(dict(key=s,gold=t,orig=t,chapter=f['chapter'],state=f"Subject: {f.get('subject','')}\nFact: {f.get('fact','')}\nSource quote: {f.get('source_quote','') or '(none)'}"))
json.dump(rows,open('ctrl.json','w'),indent=1); print(len(pick),'entities',len(rows),'facts',collections.Counter(t for t,_ in pick))
