import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8002);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
base=f'http://127.0.0.1:{a.port}'
def post(path,body,timeout=900):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=timeout))
models=json.load(urllib.request.urlopen(base+'/v1/models',timeout=10));model=models['data'][0]['id']
results={'model':model,'models':models,'started':time.time(),'checks':[]}
def save(): a.output.write_text(json.dumps(results,indent=2)+'\n')
for prompt,expected in [
    ('What is 17 times 19? Answer with only the number.','323'),
    ('What is 144 divided by 12? Answer with only the number.','12'),
    ('Copy exactly this string, with no other text: e71b04c9a28f63d5','e71b04c9a28f63d5'),
    ('Copy exactly this string, with no other text: 9d60a2f7bc814e35','9d60a2f7bc814e35'),
    ('Copy exactly this string, with no other text: 43f1c8e09ab672d5','43f1c8e09ab672d5'),
]:
    body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':96,'chat_template_kwargs':{'enable_thinking':False}}
    start=time.monotonic();r=post('/v1/chat/completions',body);text=r['choices'][0]['message'].get('content') or ''
    row={'prompt':prompt,'expected':expected,'actual':text,'passed':text.strip()==expected,'seconds':time.monotonic()-start,'response':r}
    results['checks'].append(row);save();print('check',row['passed'],round(row['seconds'],2),repr(text),flush=True)
# A prompt around 4K tokens; record the actual token count from server usage.
filler='The observatory records wind, temperature, and rainfall every morning. '
prompt=filler*300+'\nExplain, in detail, how to design a reliable weather station and validate its measurements. Give at least twenty numbered points.'
body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False},'ignore_eos':True,'cache_prompt':False}
for label in ['first','repeat']:
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.monotonic();first=None;chunks=[];content=[];reasoning=[]
    with urllib.request.urlopen(req,timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '): continue
            data=line[6:].strip()
            if data==b'[DONE]': break
            event=json.loads(data);chunks.append(event)
            for choice in event.get('choices',[]):
                d=choice.get('delta',{})
                if d.get('content') or d.get('reasoning_content') or d.get('reasoning'):
                    if first is None:first=time.monotonic()-start
                content.append(d.get('content') or '')
                reasoning.append(d.get('reasoning_content') or d.get('reasoning') or '')
    elapsed=time.monotonic()-start
    row={'label':label,'seconds':elapsed,'first_token_seconds':first,'content':''.join(content),'reasoning':''.join(reasoning),'events':chunks}
    results.setdefault('benchmarks',[]).append(row);save()
    usage=next((e['usage'] for e in reversed(chunks) if e.get('usage')),None)
    print('benchmark',label,'seconds',round(elapsed,2),'first_token',first,'usage',usage,flush=True)
results['finished']=time.time();results['passed']=all(r['passed'] for r in results['checks']);save()
raise SystemExit(0 if results['passed'] else 1)
