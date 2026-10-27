import json, urllib.request, time, random, uuid, subprocess, pathlib, hashlib
D=pathlib.Path(__file__).resolve().parent
BASE='http://192.168.1.147:8001'
MODEL='DeepSeek-v4.1-Flash-EXL3'
def post(path,data,timeout=240):
    return urllib.request.urlopen(urllib.request.Request(BASE+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=timeout)
def count(messages):
    with post('/tokenize',{'model':MODEL,'messages':messages,'chat_template_kwargs':{'enable_thinking':False}}) as r:
        return json.load(r)['count']
rng=random.Random(1017600)
words='amber bridge cedar delta engine forest garden harbor island jacket kettle lantern meadow nickel orchard pebble quartz river silver timber valley willow yellow zephyr'.split()
rows=['Record %06d: %s.\n'%(i,' '.join(rng.choices(words,k=28))) for i in range(24000)]
expected={f'CHECKPOINT_{i}':uuid.uuid4().hex[:16].upper() for i in range(1,6)}
nonce=uuid.uuid4().hex
head='Archive test '+nonce+'. Read the archive and remember the five CHECKPOINT values.\n'
tail='\nReturn only a JSON object mapping CHECKPOINT_1 through CHECKPOINT_5 to their exact values from the archive.\n'
def messages(n,pad=0):
    a=rows[:n].copy()
    for i,depth in enumerate((.05,.25,.5,.75,.95),1):
        idx=int(n*depth)
        a[idx]='The authoritative CHECKPOINT_%d value is %s.\n'%(i,expected[f'CHECKPOINT_{i}'])+a[idx]
    return [{'role':'user','content':head+''.join(a)+' x'*pad+tail}]
target=599488
lo,hi=0,len(rows)
while lo+1<hi:
    mid=(lo+hi)//2
    c=count(messages(mid))
    print('calibrate',mid,c,flush=True)
    if c<=target:lo=mid
    else:hi=mid
m=messages(lo)
c=count(m)
pad=target-c
for _ in range(8):
    m=messages(lo,pad); c=count(m)
    if c==target:break
    pad+=target-c
assert c==target,(c,target)
body={'model':MODEL,'messages':m,'temperature':0,'max_tokens':512,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
raw=json.dumps(body).encode()
(D/'request.json').write_bytes(raw)
meta={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'prompt_tokens_tokenize':c,'reserved_output_tokens':512,'rows':lo,'padding_tokens':pad,'expected':expected,'needle_depths':[.05,.25,.5,.75,.95],'request_sha256':hashlib.sha256(raw).hexdigest()}
(D/'metadata.json').write_text(json.dumps(meta,indent=2))
monitors=[]
for host in ('spark','spark2'):
    f=open(D/(host+'-memory.tsv'),'w')
    command="while true; do date -u +%s; awk '/^(MemAvailable|SwapFree|SwapTotal):/ {print $1,$2}' /proc/meminfo; sleep 3; done"
    monitors.append((subprocess.Popen(['ssh',host,command],stdout=f,stderr=subprocess.STDOUT),f))
start=time.monotonic(); result={'expected':expected,'prompt_tokens_tokenize':c}; answer=''; reasoning=''; ttft=None
print('START full context',json.dumps(meta),flush=True)
try:
    with urllib.request.urlopen(urllib.request.Request(BASE+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'}),timeout=3600) as response, open(D/'events.jsonl','w') as events:
        for line in response:
            line=line.decode().strip()
            if not line.startswith('data: '):continue
            data=line[6:]
            if data=='[DONE]':break
            event=json.loads(data); events.write(json.dumps(event)+'\n'); events.flush()
            if event.get('usage'):result['usage']=event['usage']
            if event.get('error'):result['error']=event['error']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{}); text=delta.get('content') or ''; reason=delta.get('reasoning_content') or delta.get('reasoning') or ''
                if (text or reason) and ttft is None:
                    ttft=time.monotonic()-start; print('TTFT',ttft,flush=True)
                answer+=text;reasoning+=reason
                if choice.get('finish_reason'):result['finish_reason']=choice['finish_reason']
    result['answer']=answer;result['reasoning']=reasoning
    try:
        decoded=json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
        result['retrieval_matches']={k:decoded.get(k)==v for k,v in expected.items()}
    except Exception as e:result['parse_error']=str(e)
except Exception as e:
    result['error']=repr(e)
finally:
    result['elapsed_seconds']=time.monotonic()-start;result['ttft_seconds']=ttft
    for proc,f in monitors:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill()
        f.close()
    for host in ('spark','spark2'):
        vals=[int(line.split()[1]) for line in (D/(host+'-memory.tsv')).read_text().splitlines() if line.startswith('MemAvailable:')]
        if vals:result[host+'_min_available_gib']=min(vals)/1024**2
    (D/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
