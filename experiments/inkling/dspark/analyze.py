"""Export measured speed and API checks without prompts or generated text."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def load(path):
    return json.loads(path.read_text())


def structured_prefix(row):
    """Check each fully emitted record even when the 512-token budget cuts JSON."""
    prompt=row['request']['messages'][-1]['content']
    pattern=r'^(Meria \d+) lives in (Harbor \d+), carries (copper key \d+), priced at (\d+) copper\.$'
    expected=[{'name':name,'home':home,'item':item,'price':int(price)}
              for name,home,item,price in re.findall(pattern,prompt,re.M)]
    match=re.search(r'"records"\s*:\s*\[',row['answer'])
    actual=[]
    if match:
        cursor=match.end();decoder=json.JSONDecoder()
        while cursor<len(row['answer']):
            while cursor<len(row['answer']) and row['answer'][cursor] in ' \n\r\t,':cursor+=1
            if cursor>=len(row['answer']) or row['answer'][cursor]==']':break
            try:value,offset=decoder.raw_decode(row['answer'],cursor)
            except json.JSONDecodeError:break
            actual.append(value);cursor=offset
    return {'fully_emitted_records':len(actual),'expected_total_records':len(expected),
            'fully_emitted_prefix_matches_source':actual==expected[:len(actual)] if actual else None}


def summarize(path):
    protocol = load(path/'protocol.json');progress = load(path/'progress.json')
    assert progress['passed'] and progress['stage']=='complete'
    expected = {f'{case}-{repeat}.json' for case in ('prose','code','structured')
                for repeat in range(protocol['rounds'])}
    rows = {name:load(path/name) for name in sorted(expected)}
    assert progress['completed']==len(rows)
    assert all(r['usage']['completion_tokens']>0 for r in rows.values())
    samples = [{
        'id':name.removesuffix('.json'), 'seconds':r['seconds'],
        'first_text_event_seconds':r['first_text_event_seconds'],
        'prompt_tokens':r['usage']['prompt_tokens'], 'output_tokens':r['usage']['completion_tokens'],
        'completion_tokens_per_wall_second':r['completion_tokens_per_wall_second'],
        'finish_reasons':r['finish_reasons'], 'timings':r['timings'],
        'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),
        'answer_characters':len(r['answer']),
        'reasoning_characters':len(r['reasoning']),
    } for name,r in rows.items()]
    smokes = load(path/'smokes.json')
    proposed=sum(r['timings'].get('draft_n',0) for r in rows.values())
    accepted=sum(r['timings'].get('draft_n_accepted',0) for r in rows.values())
    return {'protocol':protocol, 'samples':samples,
        'draft_tokens_generated':proposed, 'draft_tokens_accepted':accepted,
        'draft_acceptance_rate':accepted/proposed if proposed else None,
        'structured_prefix_checks':{name:structured_prefix(row) for name,row in rows.items() if name.startswith('structured-')},
        'median_completion_tokens_per_wall_second':statistics.median(r['completion_tokens_per_wall_second'] for r in rows.values()),
        'aggregate_completion_tokens_per_wall_second':sum(r['usage']['completion_tokens'] for r in rows.values())/sum(r['seconds'] for r in rows.values()),
        'median_request_seconds':statistics.median(r['seconds'] for r in rows.values()),
        'median_first_text_event_seconds':statistics.median(r['first_text_event_seconds'] for r in rows.values()),
        'per_prompt_median_rate':{case:statistics.median(r['completion_tokens_per_wall_second']
            for name,r in rows.items() if name.startswith(case)) for case in ('prose','code','structured')},
        'smokes':{name:{'passed':r['passed'],'seconds':r['seconds']} for name,r in smokes.items()},
        'smoke_passed':all(r['passed'] for r in smokes.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    arms = {name:summarize(a.root/('bench-'+name+'-20261121')) for name in ('vllm','off','dspark-3','dspark-7')}
    assert len({v['protocol']['fixture_sha256'] for v in arms.values()})==1
    assert len({v['protocol']['runner_sha256'] for v in arms.values()})==1
    launches={name:load(a.root/('inkling-gguf-'+name+'-start.json')) for name in ('off','dspark-3','dspark-7')}
    assert all(v['passed'] for v in launches.values())
    prefixes=[]
    for name,launch in launches.items():
        command=launch['command']
        boundary=command.index('--spec-type' if name=='off' else '--model-draft')
        prefixes.append(command[:boundary])
    assert all(prefix==prefixes[0] for prefix in prefixes)
    assert all(v['command'][v['command'].index('--ctx-size')+1]=='65536' for v in launches.values())
    for name,arm in arms.items():
        arm['median_rate_vs_vllm'] = arm['median_completion_tokens_per_wall_second']/arms['vllm']['median_completion_tokens_per_wall_second']
        if name!='vllm':
            arm['median_rate_vs_gguf_off'] = arm['median_completion_tokens_per_wall_second']/arms['off']['median_completion_tokens_per_wall_second']
    result={'experiment':'Inkling Mini Unsloth DSpark speed and API trial',
        'scope':'Separate from AtlasKV and earlier factual/agent quality benchmarks.',
        'complete':True, 'arms':arms,
        'gguf_launches':{name:{k:v[k] for k in ('profile','command','launcher_sha256','startup_seconds')} for name,v in launches.items()},
        'limits':['GGUF off/on uses the same quantized target; vLLM uses a different precision and runtime.',
                  'Two repetitions of three synthetic prompts on a shared device; no statistical confidence interval.',
                  'Speed probes are capped at 512 output tokens; four short API/synthetic checks provide limited correctness coverage.',
                  'Completion tokens include reasoning. All GGUF speed probes exhausted their budget before visible answer text; these are not completed-answer timings.',
                  'A later live template diagnostic found that top-level reasoning_effort none renders Inkling effort 0.9 in this pinned runtime; these original probes do not use corrected no-thinking controls.',
                  'The vLLM baseline ran during a CPU-limited download; the GGUF off/on comparison is the cleaner ablation.',
                  'SSE event gaps are not per-token ITL.'],
        'gguf_memory':{name:{k:v for k,v in load(a.root/('inkling-gguf-'+name+'-memory.json')).items()
                            if k in ('samples','min_host_available_bytes','max_swap_used_bytes',
                                     'max_process_rss_bytes','floor_tripped')}
                       for name in launches},
        'assets':load(a.root/'download-receipt.json')}
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
