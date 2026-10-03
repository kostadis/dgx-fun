# Per-fact typing test. Facts = every extracted fact about a GM-merged entity (any member), as qwen typed it
# in the extraction pass. Gold entity type = the GM's primary. Clef sees the fact, its subject and its source quote.
import json,collections,random
from facts_index import index,C
a=index('per_chapter'); b=index('per_chapter_rerun')
G=json.load(open(f'{C}/docs/ensemble/.type_merge_decisions.json'))['groups']
rows=[]
for g in G:
    for r in g['resolution']:
        if r['status']!='merged': continue
        gold=r['primary'].split('_')[0]
        for m in r['members']:
            k=tuple(m[:-3].split('_',1)); fs=a.get(k) or b.get(k) or []
            for f in fs:
                rows.append(dict(key=g['key'],gold=gold,orig=f.get('type'),passes=f.get('passes'),chapter=f['chapter'],
                    state=f"Subject: {f.get('subject','')}\nFact: {f.get('fact','')}\nSource quote: {f.get('source_quote','') or '(none)'}"))
random.seed(1); random.shuffle(rows)
json.dump(rows,open('facts.json','w'),indent=1)
c=collections.Counter(); 
for r in rows: c['match' if r['orig']==r['gold'] else 'evt/thread/date' if r['orig'] in('event','thread','date') else 'other entity type'] +=1
print(len(rows),'facts across',len({r['key'] for r in rows}),'entities; qwen extraction types vs GM:',c)
