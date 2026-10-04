import json,glob,re,os,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
C='/home/kostadis/out-of-the-abyss/out-of-the-abyss/summaries'
def cues(path):
    out=[]
    for blk in re.split(r'\n\s*\n',open(path,errors='ignore').read()):
        ls=[l for l in blk.strip().split('\n') if l and '-->' not in l and not l.strip().isdigit() and l.strip()!='WEBVTT']
        if ls: out.append(' '.join(ls))
    return out
cache={}
def tape(sess):
    if sess not in cache:
        ps=[p for p in glob.glob(f'{C}/{sess}/*.vtt') if '.cleaned' not in p and '.speakers' not in p and 'RAW' not in p]
        ps.sort(key=lambda p:(os.path.basename(p)!='transcript.vtt',p)); cache[sess]=cues(ps[0]) if ps else []
    return cache[sess]
rows=json.load(open('triage_labels.json'))
for r in rows:
    cs=tape(r['session']); t=r['token'].lower()
    hits=[i for i,c in enumerate(cs) if re.search(r'\b'+re.escape(t)+r'\b',c.lower())] or [i for i,c in enumerate(cs) if t in c.lower()]
    lines=[]
    for i in hits[:3]: lines.append(' / '.join(cs[max(0,i-1):i+2]))
    r['n']=len(hits)
    r['state']=f"Capitalised token found in a D&D session transcript: {r['token']}\nOccurrences: {len(hits)}\n\nWhere it appears (up to 3 places, with the neighbouring lines):\n"+('\n'.join('- '+l for l in lines) or '(not found)')
CRIT={"filler":"An ordinary English word, interjection, number, or sentence-start capital; not a name at all",
      "rules":"A D&D rules term, spell, monster type, or game mechanic",
      "realworld":"A real-world person, company, product, place, or another game or film; table chatter",
      "campaign":"A name from inside the fantasy campaign (character, place, faction, item, deity), possibly misheard by speech recognition"}
Q={"kind":{"type":"choice","instructions":"Speech recognition produced this capitalised token in a transcript of a D&D game played over video call. What kind of token is it?","criteria":CRIT}}
def post(u,b,to=300): return json.load(urllib.request.urlopen(urllib.request.Request(u,json.dumps(b).encode(),{"content-type":"application/json"}),timeout=to))
def clef(r,m):
    name,_,port=m.partition('@'); port=port or '8002'
    return post(f"http://192.168.1.121:{port}/v1/systemone",{"model":name,"state":r['state'],"questions":Q})['answers']['kind']['probabilities']
def qwen(r):
    p=r['state']+"\n\n"+Q['kind']['instructions']+" Options:\n"+'\n'.join(f'- "{k}": {v}' for k,v in CRIT.items())+'\n\nReply with JSON only: {"answer": "<key>"}'
    t=post("http://192.168.1.147:8001/v1/chat/completions",{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":60,"chat_template_kwargs":{"enable_thinking":False}})['choices'][0]['message']['content'] or ''
    m=re.search(r'"answer"\s*:\s*"(\w+)"',t); a=m.group(1) if m else None
    return {k:(1.0 if k==a else 0.0) for k in CRIT}
who=sys.argv[1]
def one(r):
    t=time.time(); p=qwen(r) if who=='qwen' else clef(r,who)
    return {**{k:r[k] for k in('session','token','label','n')},'p':p,'ms':(time.time()-t)*1000}
out=list(ThreadPoolExecutor(1 if who!='qwen' else 8).map(one,rows))
json.dump(out,open(f"tri_{who.split('@')[0]}.json",'w'),indent=1); print(who,'done',len(out))
