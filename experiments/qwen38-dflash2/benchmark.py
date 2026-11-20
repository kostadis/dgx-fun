import argparse,json,time,urllib.error,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bench-only',action='store_true');a=p.parse_args();base=f'http://127.0.0.1:{a.port}'
models=json.load(urllib.request.urlopen(base+'/v1/models',timeout=10));model=models['data'][0]['id']
result={'label':a.label,'model':model,'started':time.time(),'sanity':[],'benchmarks':[]}
def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
def chat(prompt,n):
 body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':604,'max_tokens':n,'stream':True,'stream_options':{'include_usage':True},'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});events=[];pieces=[];first=None;start=time.monotonic()
 try:
  with urllib.request.urlopen(req,timeout=1200) as response:
   for line in response:
    if not line.startswith(b'data: '):continue
    raw=line[6:].strip()
    if raw==b'[DONE]':break
    e=json.loads(raw);events.append(e)
    if e.get('error'):raise RuntimeError(e['error'])
    for choice in e.get('choices',[]):
     d=choice.get('delta',{});content=d.get('content') or ''
     if content and first is None:first=time.monotonic()-start
     pieces.append(content)
 except urllib.error.HTTPError as error:raise RuntimeError(error.read().decode()) from error
 elapsed=time.monotonic()-start;usage=next((e['usage'] for e in reversed(events) if e.get('usage')),None);timings=next((e['timings'] for e in reversed(events) if e.get('timings')),None)
 return {'prompt':prompt,'max_tokens':n,'text':''.join(pieces),'seconds':elapsed,'first_token_seconds':first,'usage':usage,'timings':timings,'events':events}
try:
 if not a.bench_only:
  for prompt,expected in [('What is 17 times 19? Answer with only the number.','323'),('What is 144 divided by 12? Answer with only the number.','12'),('Copy exactly, with no other text: a83d90f7c1426be5','a83d90f7c1426be5'),('Copy exactly, with no other text: f09c51e82b6734ad','f09c51e82b6734ad'),('Copy exactly, with no other text: 3e7b0d6c91a245f8','3e7b0d6c91a245f8')]:
   row=chat(prompt,96);row.update(expected=expected,passed=row['text'].strip()==expected);result['sanity'].append(row);save();print('sanity',row['passed'],round(row['seconds'],2),repr(row['text']),flush=True)
  result['sanity_passed']=all(x['passed'] for x in result['sanity'])
  if not result['sanity_passed']:raise RuntimeError('sanity gate failed')
 prompts=[('prose','Write a detailed technical guide to designing a reliable distributed cache. Cover eviction, consistency, cache stampedes, TTLs, observability, and failure recovery. Use at least twenty numbered paragraphs and at least 1000 words.'),('code','Write a complete Python module implementing a thread-safe LRU cache with generic typing, expiration, and statistics. Include detailed docstrings, edge-case handling, and at least ten pytest tests. Output the full code.'),('list','Produce a numbered list of the first 100 prime numbers. For every entry write its index, the prime, and one complete sentence explaining how to check whether that number is prime. Do not abbreviate the list.')]
 for name,prompt in prompts:
  row=chat(prompt,384);row['name']=name;result['benchmarks'].append(row);save();print('benchmark',name,'seconds',round(row['seconds'],2),'usage',row['usage'],'timings',row['timings'],flush=True)
 result['finished']=time.time();result['passed']=result.get('sanity_passed',True);save()
except Exception as error:
 result.update(passed=False,error=repr(error),finished=time.time());save();print('ERROR',repr(error),flush=True);raise
