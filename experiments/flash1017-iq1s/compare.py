import argparse,json,random,time,urllib.request,pathlib
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=random.Random(10171);cases=[]
for i in range(10):
 value=''.join(r.choices('0123456789abcdef',k=200))
 cases.append({'name':'hex-copy-'+str(i),'prompt':'Copy the following string exactly. Output only the string, without quotes or formatting:\n'+value,'expected':value,'max_tokens':256})
for i,(q,ans) in enumerate([('What is 17 times 19? Reply only with the integer.','323'),('What is 123 times 456? Reply only with the integer.','56088')]):cases.append({'name':'arithmetic-'+str(i),'prompt':q,'expected':ans,'max_tokens':32})
cases.append({'name':'prose','prompt':'Write a 180-word scene about a ferry captain discovering an unexpected passenger. Keep it grounded, with dialogue and a clear ending.','max_tokens':400})
cases.append({'name':'coding','prompt':'Write a Python function merge_intervals(intervals) that merges overlapping closed intervals, handles empty input, and does not mutate its input. Include three assert-based examples. Output only Python code.','max_tokens':512})
out={'url':a.url,'model':a.model,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cases':[]}
for c in cases:
 body={'model':a.model,'messages':[{'role':'user','content':c['prompt']}],'temperature':0,'max_tokens':c['max_tokens'],'chat_template_kwargs':{'enable_thinking':False}}
 t=time.monotonic()
 try:
  with urllib.request.urlopen(urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as resp:data=json.load(resp)
  answer=data['choices'][0]['message']['content'];x={**c,'elapsed_seconds':time.monotonic()-t,'answer':answer,'usage':data.get('usage'),'timings':data.get('timings'),'finish_reason':data['choices'][0].get('finish_reason')}
  if 'expected' in c:x['exact_match']=answer.strip()==c['expected']
 except Exception as e:x={**c,'error':str(e),'elapsed_seconds':time.monotonic()-t}
 out['cases'].append(x);pathlib.Path(a.output).write_text(json.dumps(out,indent=2));print(c['name'],round(x['elapsed_seconds'],2),x.get('exact_match',x.get('error','generated')),flush=True)
