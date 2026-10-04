import json,sys,time,urllib.request
model,arm=sys.argv[1],sys.argv[2]; rows=json.load(open(f'cards{arm}.json'))
Q={"accept":{"type":"noul","instructions":"A tool proposed replacing the transcript token with the proposed replacement in a D&D session transcript. Will the dungeon master approve this replacement exactly as proposed, with no change and no need to discuss it?"}}
out=[]
for r in rows:
    body=json.dumps({"model":model,"state":r['state'],"questions":Q}).encode()
    t=time.time();resp=json.load(urllib.request.urlopen(urllib.request.Request("http://192.168.1.121:8002/v1/systemone",body,{"content-type":"application/json"}),timeout=300))
    resp['answers']['chatter']={'noul':0}
    out.append({**{k:r[k] for k in('session','id','verdict','note')},'resp':resp,'ms':(time.time()-t)*1000,'tok':resp['usage']['input_tokens']})
json.dump(out,open(f'out_{model}_arm{arm}.json','w'),indent=1)
