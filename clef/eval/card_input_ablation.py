"""Rerun the Entemok card with isolated input repairs and unchanged controls.

Private prompts and responses stay in --workdir. Never overwrites frozen E1.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

import tasks as T

SESSION = '2026-01-12-chapter_43-harvesting-horror-in-the-underdark'
TARGET = 'c_entemok_moon'
SIB_START = '\n\nSecond, independent transcription at the same span:\n'
SIB_END = '\n\nProposed replacement found in canon sources:'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def variants(original, edit, complete_sibling):
    prefix, rest = original.split(SIB_START, 1)
    old_sibling, suffix = rest.split(SIB_END, 1)
    if not complete_sibling.startswith(old_sibling):
        raise ValueError('Recovered sibling must extend the exact original excerpt')
    old, new = edit['old'], edit['new']
    if edit['was'].replace(old, new) != edit['now']:
        raise ValueError('Before/after edit contains additional changes')
    states = {}
    for span, scope, sibling in itertools.product((False, True), repeat=3):
        state = original
        name = '_'.join(k for k, yes in [('span', span), ('scope', scope), ('sibling', sibling)] if yes) or 'original'
        if span:
            state = 'Transcript token: ' + old + '\n' + state.split('\n', 1)[1]
        if scope:
            state = state.replace('\nOccurrences on tape:',
                f"\nEdit scope: Apply only at cue {edit['cue']} in this transcript. "
                'Do not add a glossary rule. This proposal changes no other words in the cue.'
                '\nOccurrences on tape:', 1)
        if sibling:
            before, after = state.split(SIB_START, 1)
            _, tail = after.split(SIB_END, 1)
            state = before + SIB_START + complete_sibling + SIB_END + tail
        states[name] = state
    states['faithful_edit'] = states['span_scope_sibling'] + (
        f"\n\nExact sentence before this proposal:\n{edit['was']}"
        f"\n\nExact sentence after this proposal:\n{edit['now']}")
    return states


def prepare(args):
    args.workdir.mkdir(parents=True, exist_ok=True)
    if (args.workdir / 'prompts.json').exists():
        raise ValueError('Choose a new workdir')
    read = lambda p: json.loads(p.read_text())
    manifest = read(args.data / 'MANIFEST.json')
    raw = {(r['session'], r['id']): r for r in read(args.data / 'e1_cards_rawC.json')}
    saved = {(r['session'], r['id']): r for r in read(args.data / 'results/e1_rawC/gemini.json')}
    original = raw[SESSION, TARGET]['state']
    snapshot = lambda path: subprocess.check_output(
        ['git', '-C', str(args.campaign), 'show', manifest['campaign_commit'] + ':./' + path], text=True)
    mapping_text = snapshot(f'summaries/{SESSION}/spell_review/review_map.json')
    mapping = json.loads(mapping_text)[TARGET]
    assert mapping['scope'] == 'cue' and len(mapping['edits']) == 1
    edit = mapping['edits'][0]
    assert edit['cue'] == 627
    old_sibling = original.split(SIB_START, 1)[1].split(SIB_END, 1)[0]
    query = old_sibling.split('query:   ', 1)[1].split('\nsibling:', 1)[0]
    header = old_sibling.split('\nsibling: ', 1)[1].split('\n', 1)[0]
    source_path = header.rsplit(' (', 1)[0]
    relative = str(Path(source_path).relative_to(args.campaign))
    transcript = snapshot(relative)
    frozen_path = args.workdir / 'sibling-snapshot.vtt'
    frozen_path.write_text(transcript)
    module_spec = importlib.util.spec_from_file_location('sibling_context_ablation', args.sibling_helper)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    utterances = module.utterances(frozen_path)
    match_line = int(re.search(r'\bline (\d+)', old_sibling)[1])
    position = next(i for i, utterance in enumerate(utterances) if utterance[0] == match_line)
    # The original request already located the match. Recover its full window,
    # rather than repeating a global search over the entire recording.
    match = module.best_matches(utterances[position:position + 12], query, 1)[0]
    assert match['line'] == match_line
    speakers = f" [{', '.join(match['speakers'])}]" if match['speakers'] else ''
    complete = (f"query:   {query}\nsibling: {header}\n\n"
                f"  score {match['score']:.3f}  line {match['line']}{speakers}\n    {match['text']}")
    assert complete.startswith(old_sibling)
    states = variants(original, edit, complete)
    for key in ('g_entimox_boon__entemoch_s_boon', 'g_entomox_boon__entemoch_s_boon', 'g_zuktamoy__zuggtmoy'):
        states['control:' + key] = raw[SESSION, key]['state']
    baseline = {name: saved[SESSION, name.split(':', 1)[1] if name.startswith('control:') else TARGET]
                ['answers']['accept']['noul'] for name in states}
    prompts = {'model': 'gemini-decision-1', 'questions': T.Q_E1_RAW, 'states': states,
               'previous_scores': baseline,
               'provenance': {'campaign_commit': manifest['campaign_commit'],
                   'original_state_sha256': digest(original), 'mapping_sha256': digest(mapping_text),
                   'sibling_transcript_sha256': digest(transcript),
                   'sibling_helper_sha256': hashlib.sha256(args.sibling_helper.read_bytes()).hexdigest(),
                   'sibling_match_line': match['line'], 'sibling_match_score': match['score'],
                   'old_sibling_characters': len(old_sibling), 'full_sibling_characters': len(complete)}}
    (args.workdir / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    return prompts


def run(args, prompts):
    model_name = getattr(args, 'model', 'gemini')
    result_path = args.workdir / 'responses.jsonl'
    if result_path.exists():
        raise ValueError('Response file exists; choose a new workdir')
    jobs = []
    rng = random.Random(20261112)
    for repetition in range(args.repeats):
        names = list(prompts['states']); rng.shuffle(names)
        jobs.extend((repetition, name) for name in names)
    rows = []
    with result_path.open('w') as sink:
        for repetition, name in jobs:
            for attempt in range(3):
                try:
                    start = time.monotonic()
                    answers, model, usage = T.systemone(model_name, prompts['states'][name], prompts['questions'])
                    row = {'variant': name, 'repetition': repetition, 'answers': answers,
                           'served': model, 'usage': usage, 'ms': (time.monotonic() - start) * 1000}
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(5)
            sink.write(json.dumps(row) + '\n'); sink.flush(); rows.append(row)
            print(f"{repetition + 1}/{args.repeats} {name}: {answers['accept']['noul']:.6f}", flush=True)
    report = {'model': T.SYSTEMONE[model_name][1], 'reference_model': prompts['model'], 'repetitions_per_variant': args.repeats,
              'requests': len(rows), 'provenance': prompts['provenance'], 'variants': {}}
    for name, state in prompts['states'].items():
        values = [r['answers']['accept']['noul'] for r in rows if r['variant'] == name]
        report['variants'][name] = {'scores': values, 'median': statistics.median(values),
            'previous_score': prompts['previous_scores'][name], 'input_sha256': digest(state),
            'crosses_0_90_in_all_repeats': all(v >= .9 for v in values)}
    report['recorded_input_tokens'] = sum(r['usage']['input_tokens'] for r in rows)
    (args.workdir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path.home() / 'data/decision-eval/v1')
    parser.add_argument('--campaign', type=Path, default=Path.home() / 'out-of-the-abyss/out-of-the-abyss')
    parser.add_argument('--sibling-helper', type=Path, default=Path.home() / '.claude/skills/vtt-spell-pass/sibling_context.py')
    parser.add_argument('--report', type=Path, default=Path('/tmp/card-input-ablation-summary.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--model', choices=('gemini', 'clef', 'clef-flash'), default='gemini')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1: parser.error('--repeats must be positive')
    prompts = prepare(args)
    print('Prepared', len(prompts['states']), 'variants and controls', flush=True)
    if not args.prepare_only:
        run(args, prompts)


if __name__ == '__main__':
    main()
