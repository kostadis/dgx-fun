import json,glob,re,html
rows=[]
for f in sorted(glob.glob('/home/kostadis/out-of-the-abyss/out-of-the-abyss/summaries/*/spell_review/decisions.json')):
    d=json.load(open(f)); items={i['id']:i for i in json.load(open(f.replace('decisions.json','review_items.json')))['items']}
    for k,v in d['decisions'].items():
        it=items.get(k)
        if not it: continue
        strip=lambda s: html.unescape(re.sub(r'<[^>]+>','',s or '')).strip()
        ev=strip(it.get('ev',''))
        rows.append(dict(session=f.split('/')[-3],id=k,verdict=v,note=d['notes'].get(k,''),
            title=strip(it.get('t')),proposal=strip(it.get('y'))[:1500],evidence=ev[:1500]))
json.dump(rows,open('cards.json','w'),indent=1)
import collections;print(len(rows),collections.Counter(r['verdict'] for r in rows))
print(json.dumps(rows[40],indent=1)[:2500])
