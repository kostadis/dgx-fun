import json,sys,time,re,urllib.request
from concurrent.futures import ThreadPoolExecutor
CRIT={"npc":"A specific named individual person or intelligent being (including a named, talking creature or sentient being)",
      "monster":"A kind of creature or an unnamed hostile creature encountered in play",
      "faction":"An organisation, house, family, guild, cult, government or group of people",
      "location":"A place: a city, settlement, building, room, region or landmark",
      "object":"An item, weapon, artifact, book or other physical thing",
      "event":"Something that happened: an action, fight, meeting or occurrence",
      "thread":"An open plot thread, goal, mystery, promise or unresolved question",
      "date":"A date, time, duration or calendar reference"}
INS="This fact was extracted from a D&D campaign's session records. Which type should the fact be filed under? Choose by what the subject is (npc, monster, faction, location, object), unless the fact is really about an event, an open plot thread, or a date."
def post(u,b): return json.load(urllib.request.urlopen(urllib.request.Request(u,json.dumps(b).encode(),{"content-type":"application/json"}),timeout=300))
def clef(r,m): return post("http://192.168.1.121:8002/v1/systemone",{"model":m,"state":r['state'],"questions":{"kind":{"type":"choice","instructions":INS,"criteria":CRIT}}})['answers']['kind']['probabilities']
def qwen(r):
    p=r['state']+"\n\n"+INS+" Options:\n"+'\n'.join(f'- "{k}": {v}' for k,v in CRIT.items())+'\n\nReply with JSON only: {"answer": "<key>"}'
    t=post("http://192.168.1.147:8001/v1/chat/completions",{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":60,"chat_template_kwargs":{"enable_thinking":False}})['choices'][0]['message']['content'] or ''
    m=re.search(r'"answer"\s*:\s*"(\w+)"',t); a=m.group(1) if m else None
    return {k:float(k==a) for k in CRIT}
src,who=sys.argv[1],sys.argv[2]; rows=json.load(open(src))
def one(r):
    t=time.time(); p=qwen(r) if who=='qwen' else clef(r,who)
    return {**{k:v for k,v in r.items() if k!='state'},'p':p,'pred':max(p,key=p.get),'ms':(time.time()-t)*1000}
out=list(ThreadPoolExecutor(8 if who=='qwen' else 1).map(one,rows))
json.dump(out,open(f"{src[:-5]}_{who}.json",'w'),indent=1); print(who,len(out))
