"""Frozen questions and a sequential agent over a completed local CROW wiki."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import time
import traceback
from llmw2.agents.steps import CROW
from runtime import Capture, QUESTION_SYSTEM, REPO, TRIAL, SEQ, load, save, sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
assert root.is_relative_to(TRIAL) and load(root/'ingest-progress.json')['passed']
assert not (root/'query-progress.json').exists(), 'Retain previous attempt; use separate continuation when needed'
state={'stage':'preflight','passed':False,'started_unix':time.time()};save(root/'query-progress.json',state)
capture=Capture(root); note_text={}; reader_system=QUESTION_SYSTEM


def answer_step(ctx, question, notes):
    supplied=[{'path':n.rel,'id':n.id,'text':n.full_text(),'body':n.main_body(),'frontmatter':n.frontmatter} for n in notes]
    note_text.clear();note_text.update({n['path']:n for n in supplied})
    evidence='\n\n'.join('['+n['path']+']\n'+n['text'] for n in supplied)
    answer=capture.complete([{'role':'system','content':reader_system},
        {'role':'user','content':'Question: '+question+'\n\nRetrieved evidence:\n'+evidence}],cap=4096)
    return answer


def fresh(name,rerank=True,controlled=True):
    target=root/name/'wiki';assert not target.exists()
    shutil.copytree(root/'wiki',target)
    steps=replace(CROW,name='crow-controlled-answer',answer=answer_step) if controlled else None
    wiki=capture.wiki(target,steps=steps,rerank=rerank,cap=4096)
    wiki.cfg.crow.k=8  # The custom-provider preset otherwise silently chooses five.
    return wiki


def audit_original():
    protected=load(root/'protected-inputs.json')
    mismatches=[n for n,h in protected.items() if sha(TRIAL/n)!=h]
    save(root/'protected-input-audit.json',{'files':len(protected),'unchanged':not mismatches,'mismatches':mismatches})
    assert not mismatches


try:
    capture.check('before-query')
    capture.phase='index-prepare'
    tick=time.monotonic()
    with capture.wiki(root/'wiki') as wiki:
        wiki.search('Adventure module', k=1)
    save(root/'index-prepare.json', {'seconds':time.monotonic()-tick, 'query':'Adventure module', 'benchmark_questions_used':False})
    snapshot={str(p.relative_to(REPO)):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='analyze.py'}
    save(root/'query-source-sha256.json',snapshot)
    original_md={str(p.relative_to(root/'wiki')):sha(p) for p in (root/'wiki').rglob('*.md')}
    save(root/'wiki-source-sha256.json',original_md)
    cases=load(TRIAL/'frozen/cases.json');assert len(cases)==48
    question=load(TRIAL/'sonnet-sequential-20261121/question.json')['question']
    save(root/'query-protocol.json',{'cases_sha256':sha(TRIAL/'frozen/cases.json'),
        'question_sha256':sha(TRIAL/'sonnet-sequential-20261121/question.json'),
        'factual_system':QUESTION_SYSTEM,'reader_cap':4096,'action_cap':2800,'seed':604,'temperature':0,
        'k_notes':8,'max_searches':8,'action_schema':SEQ.ACTION_SCHEMA,
        'controlled_answer_step':True,'upstream_source_unmodified':True,
        'agent_evidence':'Derived wiki notes, not the original module passages; final review uses original module.'})
    for name,rerank in [('questions-crow',True),('questions-search',False)]:
        capture.phase=name;state.update(stage=name,completed_questions=0);save(root/'query-progress.json',state)
        with fresh(name,rerank) as wiki:
            save(root/name/'config.json',wiki.cfg.model_dump(mode='json'))
            tick=time.monotonic()
            for c in cases:
                capture.phase=name+'/'+c['id'];note_text.clear();offset=len(capture.rows);start=time.monotonic()
                result=wiki.ask(c['question']);seconds=time.monotonic()-start
                row={'id':c['id'],'question':c['question'],'seconds':seconds,'result':result.to_dict(),
                    'notes_supplied':list(note_text.values()),'calls':[r['id'] for r in capture.rows[offset:]]}
                save(root/name/'recall'/(c['id']+'.json'),row)
                state.update(completed_questions=len(list((root/name/'recall').glob('*.json'))),question=c['id'])
                save(root/'query-progress.json',state);print(name,c['id'],round(seconds,2),flush=True)
            save(root/name/'timing.json',{'query_loop_seconds':time.monotonic()-tick})
        assert all(sha(root/name/'wiki'/n)==h for n,h in original_md.items()),'Query changed wiki text'
        capture.check('after-'+name)

    # Free lexical-selection diagnostic: no reader/classifier calls, same k=8.
    capture.phase='bm25-selection';state['stage']='bm25-selection';save(root/'query-progress.json',state)
    with capture.wiki(root/'wiki',rerank=False,embedding=False) as wiki:
        out=[]
        for c in cases:
            tick=time.monotonic();matches=wiki.search(c['question'],k=8)
            out.append({'id':c['id'],'seconds':time.monotonic()-tick,'matches':
                        [m.to_dict() if hasattr(m,'to_dict') else str(m) for m in matches]})
        save(root/'bm25-selection.json',out)

    agent_client_start=time.monotonic()
    capture.phase='agent';state.update(stage='agent',searches=0);save(root/'query-progress.json',state)
    reader_system=('Answer this research question using only retrieved evidence. Identify missing evidence. '
                   'Distinguish factual support from interpretation. Keep the answer under 350 words.')
    policy=SEQ.POLICY.replace('Cognee search tool','wiki search tool').replace('raw source passages','retrieved wiki notes').replace('raw sources','retrieved wiki notes')
    policy+='\nThe supplied wiki notes were generated from the module and may omit or misstate details. They are derived evidence, not verbatim module passages.\n'
    save(root/'agent-policy.json',{'policy':policy})
    labels={};notes={};observations=[];decisions=[];query=question;answer=None;stop=None
    with fresh('agent') as wiki:
        save(root/'agent/config.json',wiki.cfg.model_dump(mode='json'))
        started=time.monotonic();unix=time.time()
        def evidence():return '\n\n'.join('['+labels[n]+'] DERIVED WIKI NOTE '+n+'\n'+notes[n]['text'] for n in labels)
        for step in range(1,9):
            capture.phase=f'agent/search-{step:02}';note_text.clear();offset=len(capture.rows);tick=time.monotonic()
            result=wiki.ask(query)
            for path in sorted(note_text):
                if path not in labels:labels[path]=f'S{len(labels)+1:03}';notes[path]=note_text[path]
            o={'step':step,'query':query,'answer':result.text,'notes':list(note_text),
               'source_count':len(labels),'seconds':time.monotonic()-tick,'calls':[r['id'] for r in capture.rows[offset:]]}
            observations.append(o);save(root/'agent/steps'/f'{step:02}-observation.json',o)
            state.update(stage='agent',searches=step,sources=len(labels));save(root/'query-progress.json',state)
            history=[{'step':o['step'],'query':o['query'],'unverified_tool_answer':o['answer']} for o in observations]
            messages=[{'role':'system','content':policy},{'role':'user','content':'ORIGINAL QUESTION:\n'+question+
                f'\nSEARCHES COMPLETED: {step}; MAXIMUM: 8\nOBSERVATIONS IN ORDER (tool answers are unverified):\n'+
                json.dumps(history)+'\nACCUMULATED DERIVED WIKI NOTES:\n'+evidence()}]
            searched={' '.join(o['query'].lower().split()) for o in observations}
            for attempt in range(1,4):
                capture.phase=f'agent/decision-{step:02}-{attempt}'
                raw=capture.complete(messages,cap=2800,response_format=SEQ.action_response_format('json_schema'))
                try:action=SEQ.validate_action(SEQ.parse_action(raw),searched,set(labels.values()))
                except ValueError as error:
                    save(root/'agent/steps'/f'{step:02}-invalid-action-{attempt}.json',{'raw':raw,'error':str(error)})
                    messages += [{'role':'assistant','content':raw},{'role':'user','content':'Protocol correction: '+str(error)}]
                    if attempt==3:raise
                    continue
                break
            decisions.append({'after_search':step,'action':action});save(root/'agent/steps'/f'{step:02}-decision.json',decisions[-1])
            print('agent decision',step,action['action'],flush=True)
            if action['action']=='finish':answer=action['answer'];stop='agent_finished';break
            query=action['query']
        if answer is None:
            capture.phase='agent/budget-answer'
            messages=[{'role':'system','content':policy},{'role':'user','content':
                'Search budget exhausted. Return finish with no query or remaining_gaps. State unresolved uncertainty in the answer.\nQUESTION:\n'+question+'\nDERIVED WIKI NOTES:\n'+evidence()}]
            action=SEQ.validate_action(SEQ.parse_action(capture.complete(messages,cap=2800,response_format=SEQ.action_response_format('json_schema'))),set(),set(labels.values()))
            assert action['action']=='finish';answer=action['answer'];stop='search_budget_exhausted'
        save(root/'agent/result.json',{'question':question,'started_unix':unix,'seconds':time.monotonic()-started,
            'client_seconds':time.monotonic()-agent_client_start,
            'observations':observations,'decisions':decisions,'source_labels':labels,'evidence':notes,
            'answer':answer,'stop_reason':stop,'seeded_from_prior_probes':False,'manual_intervention':False})
        (root/'agent/answer.txt').write_text(answer+'\n')
    assert all(sha(root/'agent/wiki'/n)==h for n,h in original_md.items())
    audit_original();capture.check('after-query');state['passed']=True
except Exception:
    state['error']=traceback.format_exc();print(state['error'],flush=True)
finally:
    capture.close();state.update(stage='complete' if state['passed'] else 'failed',finished_unix=time.time())
    save(root/'query-progress.json',state)
raise SystemExit(0 if state['passed'] else 1)
