"""Paired local Clef/Gemini card-filter replay, including the input ablation.

Uses the exact archived Gemini prompts and frozen E1 states, with the faithful
cue-627 input substituted for both models. Never overwrites original results.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

import card_input_ablation as A
import card_filter_review as R
import tasks as T


def read(path):
    return json.loads(path.read_text())


def tier_summary(rows):
    out = {tier: {'cards': 0, 'nonapprovals': 0, 'concrete_proposals': 0,
                  'proposal_nonapprovals': 0, 'verdicts': Counter()}
           for tier in ('high', 'middle', 'cards')}
    for row in rows:
        tier = R.tier_for(row['proposal'], row['p'])
        group = out[tier]
        group['cards'] += 1; group['verdicts'][row['verdict']] += 1
        group['nonapprovals'] += row['verdict'] != 'approve'
        group['concrete_proposals'] += row['proposal']
        group['proposal_nonapprovals'] += row['proposal'] and row['verdict'] != 'approve'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.home() / 'data/decision-eval/v1')
    parser.add_argument('--gemini-ablation', type=Path, default=Path.home() / 'data/decision-eval/card-input-ablation-v1')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report_path = args.workdir / 'comparison.json'
    if report_path.exists():
        raise ValueError('Comparison already completed; choose a new directory')
    health_url = 'http://192.168.1.121:8002/health'
    start = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                health = json.load(response)
            if health.get('ready') and 'clef' in health.get('loaded', []): break
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() - start > 1200: raise TimeoutError('Clef did not become ready')
        print(f'Waiting for Clef readiness: {time.monotonic()-start:.0f}s', flush=True)
        time.sleep(10)
    (args.workdir / 'health.json').write_text(json.dumps(health, indent=2))
    smoke = {}
    for label, state in [('urgent', 'Every customer is unable to pay. Checkout is down right now.'),
                         ('routine', 'No incident. A customer asks about next year\'s optional holiday colors.')]:
        answers, mid, usage = T.systemone('clef', state,
            {'urgent': {'type': 'noul', 'instructions': 'Is there an urgent active service outage?'}})
        smoke[label] = answers
    assert smoke['urgent']['urgent']['noul'] > .5 and smoke['routine']['urgent']['noul'] < .5
    (args.workdir / 'smoke.json').write_text(json.dumps(smoke, indent=2))
    print('Clef ready; positive/negative smoke passed', flush=True)
    prompts = read(args.gemini_ablation / 'prompts.json')
    ablation_dir = args.workdir / 'ablation'; ablation_dir.mkdir(exist_ok=True)
    (ablation_dir / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    if not (ablation_dir / 'summary.json').exists():
        A.run(argparse.Namespace(workdir=ablation_dir, repeats=3, model='clef',
                                report=ablation_dir/'summary.json'), prompts)
    ablation = read(ablation_dir / 'summary.json')
    gemini_ablation = read(args.gemini_ablation / 'summary.json')
    assert A.digest(prompts['states']['faithful_edit']) == gemini_ablation['variants']['faithful_edit']['input_sha256']
    source = read(args.data / 'e1_cards_rawC.json')
    question = prompts['questions']
    states = [prompts['states']['faithful_edit'] if (r['session'],r['id']) == (A.SESSION,A.TARGET)
              else r['state'] for r in source]
    identity = hashlib.sha256(json.dumps({'states':states,'questions':question},sort_keys=True).encode()).hexdigest()
    manifest_path = args.workdir / 'inputs.json'
    manifest = {'sha256': identity, 'questions': question,
                'rows': [{'session':r['session'],'id':r['id'],'state':s} for r,s in zip(source,states)]}
    if manifest_path.exists(): assert read(manifest_path)['sha256'] == identity
    else: manifest_path.write_text(json.dumps(manifest,indent=1))
    partial = args.workdir / 'cards.partial.jsonl'
    results = []
    if partial.exists():
        results = [json.loads(line) for line in partial.read_text().splitlines(keepends=True) if line.endswith('\n')]
        for saved, row in zip(results,source):
            assert (saved['session'],saved['id']) == (row['session'],row['id'])
        partial.write_text(''.join(json.dumps(r)+'\n' for r in results))
    with partial.open('a') as sink:
        for index in range(len(results),len(source)):
            row = source[index]
            for attempt in range(3):
                try:
                    started = time.monotonic()
                    answers, mid, usage = T.systemone('clef',states[index],question)
                    assert set(answers)=={'accept'}
                    p = answers['accept']['noul']; assert 0<=p<=1
                    result = {k:row[k] for k in ('session','id','verdict')}
                    result.update(answers=answers,served=mid,usage=usage,ms=(time.monotonic()-started)*1000)
                    break
                except Exception:
                    if attempt==2: raise
                    time.sleep(5)
            sink.write(json.dumps(result)+'\n');sink.flush();results.append(result)
            if (index+1)%50==0 or index+1==len(source): print(f'Clef cards: {index+1}/{len(source)}',flush=True)
    (args.workdir / 'cards-results.json').write_text(json.dumps(results,indent=1))
    gemini = {(r['session'],r['id']):r for r in read(args.data/'results/e1_rawC/gemini.json')}
    paired = []
    for row, result in zip(source,results):
        key=row['session'],row['id']
        token=R.field(row['state'],'Transcript token');fix=R.field(row['state'],'Proposed replacement')
        g=gemini_ablation['variants']['faithful_edit']['median'] if key==(A.SESSION,A.TARGET) else gemini[key]['answers']['accept']['noul']
        paired.append({'session':key[0],'id':key[1],'verdict':row['verdict'],
                       'proposal':R.concrete_proposal(token,fix),'clef':result['answers']['accept']['noul'],'gemini':g})
    report = {'input_sha256':identity,'model':'clef','reference_model':'gemini-decision-1','rows':len(results),
              'failed_rows':0,'repeated_ablation_requests':36,'new_card_requests':len(results),
              'models':{model:tier_summary([{**r,'p':r[model]} for r in paired]) for model in ('clef','gemini')},
              'target':next(r for r in paired if r['session']==A.SESSION and r['id']==A.TARGET),
              'clef_median_ms':statistics.median(r['ms'] for r in results),
              'tier_changes':sum(R.tier_for(r['proposal'],r['clef'])!=R.tier_for(r['proposal'],r['gemini']) for r in paired),
              'ablation':ablation,
              'scope':'Same 766 frozen card states, with faithful cue-627 input substituted for both models; unchanged question. Historical evaluation, not new training.'}
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'models':report['models'],'target':report['target'],'clef_median_ms':report['clef_median_ms']},indent=2),flush=True)


if __name__=='__main__': main()
