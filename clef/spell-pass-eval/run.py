import json,sys,time,urllib.request
model=sys.argv[1]; rows=json.load(open('cards.json')); lim=int(sys.argv[2]) if len(sys.argv)>2 else None
Q={"accept":{"type":"noul","instructions":"A tool proposed this correction to a D&D session transcript. Will the dungeon master approve it exactly as written, with no change and no need to discuss it?"},
   "chatter":{"type":"noul","instructions":"Is the name in question out-of-game table chatter (a real-world person, company, product, place, or a different game) rather than something inside this D&D campaign?"}}
out=[]
for r in rows[:lim]:
    state=f"Card: {r['title']}\n\nProposal: {r['proposal']}\n\nEvidence: {r['evidence']}"
    body=json.dumps({"model":model,"state":state,"questions":Q}).encode()
    t=time.time();resp=json.load(urllib.request.urlopen(urllib.request.Request("http://192.168.1.121:8002/v1/systemone",body,{"content-type":"application/json"}),timeout=120))
    out.append({**{k:r[k] for k in('session','id','verdict','note')},'resp':resp,'ms':(time.time()-t)*1000})
    if lim: print(json.dumps(resp)[:800])
json.dump(out,open(f'out_{model}.json','w'),indent=1)
