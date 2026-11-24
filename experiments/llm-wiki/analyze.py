"""Audit and export metrics/review flags without publishing source text or answers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import time
import yaml
from runtime import REPO, TRIAL, load, save, sha


def percentile(values, f):
    v=sorted(values);i=(len(v)-1)*f;j=int(i)
    return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(i-j)


def summarize(rows):
    llm=[r for r in rows if r['kind']=='llm'];classifier=[r for r in rows if r['kind']=='classifier']
    embed=[r for r in rows if r['url'].endswith('/api/embed')]
    def usage(rows,keys):
        return {k:sum(r['response'].get('usage',{}).get(k,0) or 0 for r in rows) for k in keys}
    return {'llm_calls':len(llm),'llm_seconds':sum(r['seconds'] for r in llm),
        'llm_finish_reasons':dict(Counter(r['response'].get('choices',[{}])[0].get('finish_reason','http_error') for r in llm)),
        'llm_usage':usage(llm,['prompt_tokens','completion_tokens','total_tokens']),
        'classifier_calls':len(classifier),'classifier_seconds':sum(r['seconds'] for r in classifier),
        'classifier_usage':usage(classifier,['input_tokens','output_tokens']),
        'embedding_calls':len(embed),'embedding_seconds':sum(r['seconds'] for r in embed),
        'http_failures':sum(r['status']!=200 for r in rows)}


p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(TRIAL)
ingest=load(root/'ingest-progress.json');query=load(root/'query-progress.json');assert ingest['passed'] and query['passed']
protocol=load(root/'protocol.json');qprotocol=load(root/'query-protocol.json');before=load(root/'services-before.json')
for f in root.glob('services-*.json'):
 s=load(f);assert s['native']==before['native'] and s['clef']==before['clef']
 assert s['clef_health']['ready'] and s['guard_active'] and not s['guard_stats']['floor_tripped']
 assert s['available_bytes']>=12*1024**3 and s['guard_stats']['container_id']==s['native']['id']
for name,h in load(root/'protected-inputs.json').items():assert sha(TRIAL/name)==h
executed_hashes=load(root/'query-source-sha256.json');archived_sources=[]
for name,h in executed_hashes.items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 current=REPO/name
 if current.is_file() and sha(current)==h:continue
 # Later replay fixes must not rewrite the code provenance of the measured run.
 assert sha(root/'executed-source'/name)==h
 archived_sources.append(name)
for name,h in load(root/'wiki-source-sha256.json').items():assert sha(root/'wiki'/name)==h
for name,h in protocol['input_sha256'].items():assert sha(TRIAL/'frozen'/(name+'.json'))==h
rows=[load(f) for f in (root/'calls').glob('*.json')];byid={r['id']:r for r in rows}
embedding_digest=load(TRIAL/'current/manifest.json')['embedding_digest']
for r in rows:
 if r['url'].endswith('/api/tags'):
  assert next(m['digest'] for m in r['response']['models'] if m['name']=='qwen3-embedding:0.6b')==embedding_digest
for r in rows:
 if r['kind']=='llm':
  b=r['request'];assert b['model']=='inkling-2-mini' and b['reasoning_effort']=='none'
  assert b['bad_words']==['<|content_thinking|>'] and b['chat_template_kwargs']['enable_thinking'] is False
  assert b['temperature']==0 and b['seed']==604
  if r['status']==200:
   msg=r['response']['choices'][0]['message'];assert not(msg.get('reasoning_content') or msg.get('reasoning'))
   assert r['response']['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0)==0
 if r['kind']=='classifier':assert r['request']['model']=='clef' and r['url'].startswith('http://192.168.1.121:8002/')
 if r['url'].endswith('/api/embed'):assert r['request']['truncate'] is False and r['request']['model']=='qwen3-embedding:0.6b'

raw_owners={};documents={d['id']:d for d in load(TRIAL/'frozen/corpus.json')['documents']}
for path in (root/'wiki/raw').glob('*.md'):
 parts=path.read_text().split('---',2);meta=yaml.safe_load(parts[1]);identity=meta['resource'].removeprefix('module://')
 assert identity in documents and parts[2].strip()==documents[identity]['text'].strip()
 raw_owners['/'+str(path.relative_to(root/'wiki'))]=identity
assert set(raw_owners.values())==set(documents)
def owners(notes):
 return {raw_owners[src['resource']] for n in notes for src in n['frontmatter'].get('sources',[]) if src.get('resource') in raw_owners}
factual={};cases=load(TRIAL/'frozen/cases.json');evidence_checks=0
for arm in ['questions-crow','questions-search']:
 cfg=load(root/arm/'config.json');assert cfg['crow']['k']==8 and cfg['crow']['rerank']==(arm=='questions-crow')
 recs=[load(root/arm/'recall'/(c['id']+'.json')) for c in cases];assert len(recs)==48
 reviews=load(root/(arm+'-review.json'));assert set(reviews)=={c['id'] for c in cases}
 per=[]
 for case,r in zip(cases,recs):
  assert case['id']==r['id'] and case['question']==r['question']
  captured=[byid[k] for k in r['calls'] if byid[k]['kind']=='llm' and byid[k]['request']['messages'][0]['content']==qprotocol['factual_system']]
  if r['notes_supplied']:
   assert len(captured)==1;body=captured[0]['request'];assert body['max_tokens']==4096
   assert captured[0]['response']['choices'][0]['finish_reason']=='stop'
   text='\n\n'.join('['+n['path']+']\n'+n['text'] for n in r['notes_supplied'])
   assert body['messages'][1]['content'].endswith(text) and case['question'] in body['messages'][1]['content']
   assert not any(other['question'] in body['messages'][1]['content'] for other in cases if other['id']!=case['id'])
   evidence_checks+=len(r['notes_supplied'])
  review=reviews[case['id']];assert (review['correctness']=='source_ambiguous')==(case['id']=='q37')
  assert review['correctness'] in ['correct_and_complete','partial','incorrect','abstained','source_ambiguous']
  assert review['grounding'] in ['supported','unsupported','unassessed']
  per.append({'id':case['id'],'seconds':r['seconds'],'notes':len(r['notes_supplied']),
      'quality':{k:review[k] for k in ['correctness','grounding']},
      'required_source_documents_represented':{e['doc_id'] for e in case['evidence']}.issubset(owners(r['notes_supplied']))})
 times=[r['seconds'] for r in recs]
 factual[arm]={**summarize([r for r in rows if r['phase'].startswith(arm+'/')]),
   'source_document_hit_count':sum(r['required_source_documents_represented'] for r in per),
   'query_sum_seconds':sum(times),'p50_seconds':statistics.median(times),'p95_seconds':percentile(times,.95),
   'quality':{'adjudicable':47,'excluded':['q37'],'correct_and_complete':sum(v['correctness']=='correct_and_complete' for v in reviews.values()),
     'grounded_complete':sum(v['correctness']=='correct_and_complete' and v['grounding']=='supported' for v in reviews.values()),
     'counts':dict(Counter(v['correctness'] for v in reviews.values()))},'per_question':per}
agent=load(root/'agent/result.json');assert not agent['seeded_from_prior_probes'] and not agent['manual_intervention']
assert len(agent['observations'])<=8
acfg=load(root/'agent/config.json');assert acfg['crow']['k']==8 and acfg['crow']['rerank']
cited=set(re.findall(r'\bS\d{3}\b',agent['answer']));assert cited and cited.issubset(set(agent['source_labels'].values()))
for i,o in enumerate(agent['observations']):
 if i:assert o['query']==agent['decisions'][i-1]['action']['query']
 assert all(n in agent['evidence'] for n in o['notes'])
 for k in o['calls']:assert byid[k]['started_unix']>=agent['started_unix']
 reader=[byid[k] for k in o['calls'] if byid[k]['kind']=='llm']
 assert len(reader)==1 and reader[0]['request']['max_tokens']==4096
 supplied='\n\n'.join('['+n+']\n'+agent['evidence'][n]['text'] for n in o['notes'])
 assert reader[0]['request']['messages'][1]['content'].endswith(supplied)
 assert o['query'] in reader[0]['request']['messages'][1]['content']
for r in rows:
 if r['phase'].startswith(('agent/decision-','agent/budget-answer')) and r['kind']=='llm':
  assert r['request']['max_tokens']==2800 and r['request']['response_format']['type']=='json_schema'
note_by_path={n['path']:n for n in load(root/'notes.json')}
bm25=load(root/'bm25-selection.json');assert len(bm25)==48
assert not any(r['phase']=='bm25-selection' for r in rows), 'Lexical diagnostic must not call models'
bm25_hits=sum({e['doc_id'] for e in c['evidence']}.issubset(owners([note_by_path[m['path']] for m in r['matches']])) for c,r in zip(cases,bm25))
last=load(root/'services-final-review.json')
verification=load(root/'verification.json')
assert verification['repository_tests']['exit_code']==0 and verification['adapter_regression']['exit_code']==0
assert verification['upstream_worktree_clean']
baselines={}
for name,folder in [('stock','stock-cognee'),('adaptive','fp8-cognee')]:
 path=REPO/'experiments/inkling'/folder/'results-20261122.json';b=load(path)
 assert b['native']['image']==before['native']['image'] and b['native']['command']==before['native']['command']
 baselines[name]={'path':str(path.relative_to(REPO)),'sha256':sha(path),'factual':
    {k:b['factual'][k] for k in ['p50_seconds','p95_seconds','query_sum_seconds','usage','quality']},
    'agent':{k:b['agent'][k] for k in ['client_seconds','research_seconds','searches','llm_calls','usage','source_review']}}
export={'experiment':'Full Phandelver CROW/Clef versus saved Cognee indexes',
 'complete':True,'verified_utc':last['checked_utc'],'upstream_commit':protocol['upstream_commit'],
 'upstream_license':protocol['upstream_license'],'source_documents':13,'wiki_notes':ingest['notes'],
 'baselines':baselines,
 'all_local':True,'models':{'writer':'inkling-2-mini','reader':'inkling-2-mini','classifier':'clef','embedder':'qwen3-embedding:0.6b'},
 'embedding':{'digest':embedding_digest,'dimensions':1024,'api_truncation':False,
    'wiki_note_body_prefix_characters':3000,'rerank_passage_characters':600},
 'input_sha256':protocol['input_sha256'],'analysis_sha256':sha(__file__),
 'ingestion':{'seconds':ingest['ingest_seconds'],**summarize([r for r in rows if r['phase']=='ingest'])},
 'index_preparation_seconds':load(root/'index-prepare.json')['seconds'],'factual':factual,
 'bm25_selection':{'questions':48,'k':8,'required_source_documents_represented':bm25_hits,
    'meaning':'Source representation only; does not verify that requested facts survived into a generated note.'},
 'agent':{'research_seconds':agent['seconds'],'client_seconds':agent['client_seconds'],'searches':len(agent['observations']),'notes':len(agent['source_labels']),
    'protocol_corrections':len(list((root/'agent/steps').glob('*invalid-action*.json'))),
    'stop_reason':agent['stop_reason'],**summarize([r for r in rows if r['phase'].startswith('agent/')]),
    'source_review':load(root/'agent-review-public.json')},
 'audit':{'historical_inputs_unchanged':True,'historical_files':len(load(root/'protected-inputs.json')),
    'factual_note_context_appearances_verified':evidence_checks,'wiki_text_unchanged':True,
    'executed_source_sha256':executed_hashes,'executed_source_archive_used':archived_sources,
    'generation_controls_verified':True,'zero_reasoning_tokens':True,'services_unchanged':True},
 'verification':verification,
 'operations':{'native':before['native'],'clef':before['clef'],'final_available_bytes':last['available_bytes'],
    'guard_floor_tripped':False,'qwen_model_ids':last['qwen_model_ids']},
 'limits':['The wiki was ingested by Inkling; saved Cognee graphs were ingested by Qwen.',
   'Controlled answer step uses the shared benchmark policy rather than the native wiki answer prompt.',
   'Wiki agent evidence consists of generated notes; Cognee supplied original module chunks.',
   'One observation per question/arm and one module agent; historical baselines ran at other times.',
   'Clef custom-model thresholds were not calibrated against these questions.',
   'Codex source review, not blinded human adjudication.']}
save(a.output,export)
print(json.dumps({'ingestion':export['ingestion'],'factual':{k:{kk:vv for kk,vv in v.items() if kk!='per_question'} for k,v in factual.items()},'agent':{k:v for k,v in export['agent'].items() if k!='source_review'}},indent=2))
