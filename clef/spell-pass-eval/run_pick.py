import json,sys,time,urllib.request,re
from concurrent.futures import ThreadPoolExecutor
who=sys.argv[1]; rows=json.load(open('pick.json'))
QTEXT="Speech recognition often misspells fantasy names in this D&D session transcript. The token under review will be replaced, word for word, by the option you pick. Which option is the correct spelling of exactly the words spoken in that token? Do not pick an option that adds words the speaker did not say (for example a first name or title when only a surname was spoken)."
LEAVE="Not a mangled campaign name: leave the words as spoken (table chatter, ordinary words, rules talk)"
OTHER="A misspelled campaign name, but none of the listed options is its correct spelling"
def post(url,body,to=300):
    return json.load(urllib.request.urlopen(urllib.request.Request(url,json.dumps(body).encode(),{"content-type":"application/json"}),timeout=to))
def clef(r,model):
    crit={k:v for k,v in r['options'].items()}; crit['LEAVE']=LEAVE; crit['OTHER']=OTHER
    q={"pick":{"type":"choice","instructions":QTEXT,"criteria":crit}}
    a=post("http://192.168.1.121:8002/v1/systemone",{"model":model,"state":r['state'],"questions":q})['answers']['pick']
    return a
def qwen(r,think):
    opts='\n'.join(f'- "{k}": {v}' for k,v in r['options'].items())+f'\n- "LEAVE": {LEAVE}\n- "OTHER": {OTHER}'
    p=(f"{r['state']}\n\nSpeech recognition often mangles fantasy names in this D&D session transcript. "
       f"{QTEXT} Options:\n{opts}\n\n"
       'Reply with JSON only: {"answer": "<option key exactly>", "confidence": <0 to 1>}')
    body={"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"temperature":0,"max_tokens":4000 if think else 200,
          "chat_template_kwargs":{"enable_thinking":think}}
    t=post("http://192.168.1.147:8001/v1/chat/completions",body,600)['choices'][0]['message']['content'] or ''
    m=re.search(r'\{[^{}]*"answer"[^{}]*\}',t,re.S)
    try: j=json.loads(m.group(0)); return {"choice":j['answer'],"conf":float(j.get('confidence',0)),"raw":t[-300:]}
    except Exception: return {"choice":None,"conf":0,"raw":t[-300:]}
def one(r):
    t=time.time()
    if who.startswith('clef'): a=clef(r,who)
    else: a=qwen(r,who=='qwen-think')
    return {**{k:r[k] for k in('session','id','token','gold','gold_in_opts')},'ans':a,'ms':(time.time()-t)*1000}
n=1 if who.startswith('clef') else 6
out=list(ThreadPoolExecutor(n).map(one,rows))
json.dump(out,open(f'pick_{who}.json','w'),indent=1)
print(who,'done',len(out))
