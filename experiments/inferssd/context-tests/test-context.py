import argparse,json,random,threading,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--context',type=int,required=True);p.add_argument('--target',type=int,required=True);a=p.parse_args()
root=Path.home()/'inferssd-trial/context-tests';base='http://127.0.0.1:8002';out=root/f'{a.context}-result.json'
result={'context':a.context,'target_tokens':a.target,'started':time.time(),'memory':[]};stop=threading.Event()
def save():out.write_text(json.dumps(result,indent=2)+'\n')
def monitor():
 while not stop.is_set():
  mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','SwapFree:'))}
  result['memory'].append({'time':time.time(),**mem});stop.wait(2)
def post(path,body,timeout=30):
 q=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(q,timeout=timeout))
def messages(n):
 words='archive weather station ledger field survey meadow ridge valley harbor cloud rainfall temperature wind pressure river forest stone record morning evening north south east west season sample reading index log'.split()
 rng=random.Random(604);lines=[]
 for i in range(n):lines.append(f'Record {i:06d}: '+' '.join(rng.choices(words,k=10))+'.')
 facts={'ORBIT':'a83d90f7c1426be5','CEDAR':'f09c51e82b6734ad','MAPLE':'3e7b0d6c91a245f8','DELTA':'c4819fa620de735b','ANCHOR':'7bd02e4f9a165c83'}
 for pos,(k,v) in reversed(list(zip([0.01,0.25,0.50,0.75,0.99],facts.items()))):lines.insert(int(n*pos),f'CATALOG FACT: the value of {k} is {v}.')
 prompt='Read this archive and retain the five CATALOG FACT values. Ordinary records are unrelated background.\n\n'+'\n'.join(lines)+'\n\nReturn a JSON object with exactly these keys: ORBIT, CEDAR, MAPLE, DELTA, ANCHOR. Copy their CATALOG FACT values exactly. No explanation.'
 return [{'role':'user','content':prompt}],facts
def token_count(m):
 rendered=post('/apply-template',{'messages':m,'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})
 tokens=post('/tokenize',{'content':rendered['prompt'],'add_special':False,'parse_special':True})
 return len(tokens['tokens'])
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
 for _ in range(180):
  try:
   if json.load(urllib.request.urlopen(base+'/health',timeout=5)).get('status')=='ok':break
  except Exception:pass
  time.sleep(2)
 else:raise RuntimeError('server readiness timeout')
 props=json.load(urllib.request.urlopen(base+'/props',timeout=10));actual=props['default_generation_settings']['n_ctx'];assert actual==a.context,(actual,a.context)
 result['reported_context']=actual
 models=json.load(urllib.request.urlopen(base+'/v1/models'));model=models['data'][0]['id']
 low,high=1,7000;best=None
 while low<=high:
  n=(low+high)//2;m,facts=messages(n);count=token_count(m)
  if count<=a.target:best=(m,facts,count,n);low=n+1
  else:high=n-1
 m,facts,count,n=best
 result.update(prompt_tokens_estimated=count,records=n,expected=facts)
 (root/f'{a.context}-prompt.txt').write_text(m[0]['content'])
 body={'model':model,'messages':m,'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True},'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 result['request_started']=time.time();save();print('request',a.context,'estimated tokens',count,'records',n,flush=True)
 start=time.monotonic();first=None;events=[];pieces=[]
 with urllib.request.urlopen(req,timeout=3600) as response:
  for line in response:
   if not line.startswith(b'data: '):continue
   d=line[6:].strip()
   if d==b'[DONE]':break
   event=json.loads(d);events.append(event)
   for choice in event.get('choices',[]):
    delta=choice.get('delta',{});text=delta.get('content') or ''
    if text and first is None:
     first=time.monotonic()-start;result['first_token_seconds']=first;save();print('first token',round(first,2),flush=True)
    pieces.append(text)
 elapsed=time.monotonic()-start;text=''.join(pieces)
 parsed=json.loads(text[text.index('{'):text.rindex('}')+1]);checks={k:parsed.get(k)==v for k,v in facts.items()}
 result.update(seconds=elapsed,response=text,events=events,checks=checks,passed=all(checks.values()),finished=time.time())
 print('finished',a.context,'passed',result['passed'],'seconds',elapsed,'answer',text,flush=True)
except Exception as error:
 result.update(error=repr(error),passed=False,finished=time.time());print('ERROR',repr(error),flush=True)
finally:
 stop.set();thread.join();result['minimum_available_gib']=min(x['MemAvailable'] for x in result['memory'])/2**30;save()
raise SystemExit(0 if result['passed'] else 1)
