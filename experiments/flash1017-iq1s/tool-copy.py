import argparse,json,random,time,urllib.request,pathlib
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=random.Random(30817);out=[]
values=[''.join(r.choices('0123456789abcdef',k=200)) for _ in range(6)]
values += ['The ferry leaves at 09:42. Bring 17 copper coins, a blue notebook, and ticket AB-7294. Do not change punctuation.', 'Evelyn said, "We crossed the bridge before sunrise." The ledger records 62 crates, 4 empty barrels, and 103 silver pieces.', 'Archive entry 2026-10-21: temperature 18.5 degrees; status pending; reference XQ-314159. Preserve this text exactly.', 'A small brass key lay beneath the folded map. Three names appeared on the envelope: Mira, Tomas, and June.']
for i,value in enumerate(values):
 body={'model':a.model,'messages':[{'role':'user','content':'Call record with the exact text below as its text argument. Copy every character, without correction or commentary:\n'+value}],'tools':[{'type':'function','function':{'name':'record','description':'Record text verbatim','parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}}],'tool_choice':{'type':'function','function':{'name':'record'}},'temperature':0,'max_tokens':384,'chat_template_kwargs':{'enable_thinking':False}}
 t=time.monotonic();item={'index':i,'expected':value}
 try:
  with urllib.request.urlopen(urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300) as resp:data=json.load(resp)
  item['response']=data;calls=data['choices'][0]['message'].get('tool_calls') or [];item['exact_match']=len(calls)==1 and calls[0]['function']['name']=='record' and json.loads(calls[0]['function']['arguments']).get('text')==value
 except Exception as e:item['error']=str(e)
 item['elapsed_seconds']=time.monotonic()-t;out.append(item);pathlib.Path(a.output).write_text(json.dumps(out,indent=2));print(i,round(item['elapsed_seconds'],2),item.get('exact_match',item.get('error')),flush=True)
