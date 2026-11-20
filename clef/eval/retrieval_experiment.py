"""Paired E2 retrieval experiment; private inputs/results stay in --workdir.

prepare freezes four alternative candidate lists before any API calls. run uses
the unchanged Gemini question and state; score compares all arms with frozen E2.
Historical aliases use only approved examples from strictly earlier sessions.
"""
import argparse
from collections import Counter, defaultdict
import difflib
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

import yaml
import tasks as T

ARMS = ('wide32', 'context8', 'context32', 'history32')
SIBLING = '\n\nSecond, independent transcription at the same span:\n'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_queries(token, raw, sibling):
    """Use transcript alignment to retrieve the sibling's reading of this token.

    No labels or proposed corrections enter alignment. If the alignment is too
    broad to localize the word, return no query rather than every nearby name.
    """
    words = lambda s: re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", s)
    a, b, needle = words(raw), words(sibling), words(token)
    al, bl, nl = [w.casefold() for w in a], [w.casefold() for w in b], [w.casefold() for w in needle]
    if not nl or not bl:
        return []
    opcodes = difflib.SequenceMatcher(None, al, bl, autojunk=False).get_opcodes()
    queries = []
    for start in range(len(al) - len(nl) + 1):
        end = start + len(nl)
        if al[start:end] != nl:
            continue
        positions = []
        for tag, i, j, k, l in opcodes:
            if start < j and end > i:
                positions.extend(range(k + max(start, i) - i, k + min(end, j) - i)
                                 if tag == 'equal' else range(k, l))
        if positions and max(positions) - min(positions) < 8:
            queries.append(' '.join(b[min(positions):max(positions) + 1]))
    return list(dict.fromkeys(queries))


class Retriever:
    def __init__(self, pool, phonetic_codes):
        self.pool = pool
        self.codes = lru_cache(None)(lambda s: phonetic_codes(re.sub('[^A-Za-z]', '', s)) or set())

    @lru_cache(None)
    def similarity(self, token, name):
        # Identical ranking formula to build_pick.py; insertion order breaks ties.
        token, name = token.lower(), name.lower()
        score = difflib.SequenceMatcher(None, token, name).ratio()
        for a in token.split():
            if len(a) > 2:
                for b in name.split():
                    score = max(score, .95 * difflib.SequenceMatcher(None, a, b).ratio())
        ph = 0
        for a in [token.replace(' ', '')] + token.split():
            for b in [name.replace(' ', '')] + name.split():
                if len(a) > 2 and len(b) > 2 and self.codes(a) & self.codes(b):
                    ph = max(ph, difflib.SequenceMatcher(None, a, b).ratio())
        return max(score, .5 + .5 * ph if ph else 0)

    def rank(self, token, raw='', queries=(), history=()):
        scores = {name: self.similarity(token, name) for name in self.pool}
        for name in scores:
            for query in queries:
                scores[name] = max(scores[name], .98 * self.similarity(query, name))
            if raw and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', raw, re.I):
                scores[name] += .04
        for alias, canonical in history:
            if canonical in scores:
                value = 1.01 if token.casefold() == alias.casefold() else .99 * self.similarity(token, alias)
                scores[canonical] = max(scores[canonical], value)
        return sorted(scores, key=lambda name: -scores[name])


def snapshot_pool(campaign, commit):
    show = lambda path: subprocess.check_output(
        ['git', '-C', str(campaign), 'show', f'{commit}:./{path}'], text=True)
    pool = {}
    for entity in yaml.safe_load(show('docs/entity_registry.yaml'))['entities']:
        description = entity.get('type', '?') + (f": {entity['note']}" if entity.get('note') else '')
        for name in [entity['name']] + (entity.get('aliases') or []):
            pool.setdefault(name, description[:160])
    # Canonical column only. The wrong-form column includes the evaluation answers.
    for line in show('notes/vtt_transcription_corrections.md').splitlines():
        match = re.match(r'\|.*\|\s*\*\*(.+?)\*\*\s*\|', line)
        if match:
            pool.setdefault(match[1].strip(), 'campaign glossary name')
    paths = subprocess.check_output(['git', '-C', str(campaign), 'ls-tree', '-r', '--name-only',
                                     commit, 'docs/background'], text=True).splitlines()
    for path in paths:
        if path.endswith('inventory.md'):
            for line in show(path).splitlines():
                for name in re.findall(r'\*\*([^*]{2,60})\*\*', line):
                    pool.setdefault(name.strip(), 'published module: ' + line.strip().replace('*', '')[:140])
    return pool


def prepare(args):
    args.workdir.mkdir(parents=True, exist_ok=True)
    target = args.workdir / 'MANIFEST.json'
    if target.exists():
        raise RuntimeError('Already prepared; choose a new workdir')
    source = read(args.data / 'e2_pick.json')
    manifest = read(args.data / 'MANIFEST.json')
    assert digest(args.data / 'e2_pick.json') == manifest['files']['e2_pick']['sha256']
    pool = snapshot_pool(args.campaign, manifest['campaign_commit'])
    # Preserve authoritative baseline descriptions and tie order where present.
    for row in source:
        for name, description in row['options'].items():
            assert name in pool, f'Frozen option missing from snapshot: {name}'
            pool[name] = description
    spec = importlib.util.spec_from_file_location('retrieval_phonetics', args.phonetics)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    retrieval = Retriever(pool, module.codes)
    arms = {name: [] for name in ARMS}
    aligned = 0
    history = []
    by_date = defaultdict(list)
    for row in source:
        assert re.match(r'\d{4}-\d{2}-\d{2}', row['session'])
        by_date[row['session'][:10]].append(row)
    generated = {}
    # Freeze every candidate list before inspecting retrieval recall or calling Gemini.
    for date in sorted(by_date):
        for row in by_date[date]:
            token, state = row['token'], row['state']
            raw, sibling = state.split('Raw transcript around its first occurrence:\n', 1)[1].split(SIBLING, 1)
            queries = aligned_queries(token, raw, sibling) if not sibling.startswith('(') else []
            aligned += bool(queries)
            baseline = list(row['options'])
            wide = baseline + [name for name in retrieval.rank(token) if name not in baseline]
            context = retrieval.rank(token, raw, queries)
            learned = retrieval.rank(token, raw, queries, history)
            generated[row['session'], row['id']] = {
                'wide32': wide[:32], 'context8': context[:8],
                'context32': context[:32], 'history32': learned[:32]}
        # Training labels become available only to later dates, never this date.
        history.extend((r['token'], r['gold']) for r in by_date[date] if r['gold'] != 'LEAVE')
    for row in source:
        for arm in ARMS:
            options = {name: pool[name] for name in generated[row['session'], row['id']][arm]}
            arms[arm].append({**row, 'options': options,
                              'gold_in_opts': row['gold'] == 'LEAVE' or row['gold'] in options})
    for arm, rows in arms.items():
        (args.workdir / f'{arm}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    # Keep the full canonical pool privately, so future campaign edits cannot change the experiment.
    (args.workdir / 'pool.json').write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    metadata = {'source_sha256': digest(args.data / 'e2_pick.json'),
                'baseline_result_sha256': digest(args.data / 'results/e2/gemini.json'),
                'campaign_commit': manifest['campaign_commit'], 'pool_size': len(pool),
                'phonetics_sha256': digest(args.phonetics), 'aligned_sibling_rows': aligned,
                'history_policy': 'Approved rows from strictly earlier session dates only; no same-date or future rulings.',
                'pool_policy': 'Original campaign snapshot canonical names, registry aliases and inventory names; glossary wrong forms excluded.',
                'state_policy': 'Unchanged frozen E2 state and question; only options vary.',
                'arms': {arm: {'sha256': digest(args.workdir / f'{arm}.json'),
                               'rows': len(rows), 'name_recall': sum(r['gold'] != 'LEAVE' and r['gold_in_opts'] for r in rows)}
                         for arm, rows in arms.items()}}
    target.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def run(args):
    manifest = read(args.workdir / 'MANIFEST.json')
    # Reuse the authorized API adapter and validate access before creating results.
    T.systemone('gemini', 'The mug arrived cracked.',
                {'damaged': {'type': 'noul', 'instructions': 'Is the mug damaged?'}})
    for arm in args.arms:
        path = args.workdir / f'{arm}.json'
        assert digest(path) == manifest['arms'][arm]['sha256']
        rows = read(path)
        out = args.workdir / f'{arm}-results.jsonl'
        complete = []
        if out.exists():
            complete = [json.loads(line) for line in out.read_text().splitlines(keepends=True) if line.endswith('\n')]
            for saved, source in zip(complete, rows):
                assert (saved['session'], saved['id']) == (source['session'], source['id'])
            assert len(complete) <= len(rows)
            out.write_text(''.join(json.dumps(r) + '\n' for r in complete))
        with out.open('a') as sink:
            for index, row in enumerate(rows[len(complete):], len(complete)):
                for attempt in range(3):
                    try:
                        start = time.monotonic()
                        answers, model, usage = T.systemone('gemini', row['state'], T.e2_questions(row))
                        answer = answers['pick']
                        expected = set(row['options']) | {'OTHER', 'LEAVE'}
                        assert set(answer['probabilities']) == expected
                        result = {'session': row['session'], 'id': row['id'], 'ans': answer,
                                  'served': model, 'usage': usage,
                                  'ms': (time.monotonic() - start) * 1000}
                        break
                    except Exception:
                        if attempt == 2:
                            raise  # Keep prefix; resume the failing row next time.
                        time.sleep(5)
                sink.write(json.dumps(result) + '\n'); sink.flush()
                if (index + 1) % 50 == 0 or index + 1 == len(rows):
                    print(f'{arm}: {index + 1}/{len(rows)}', flush=True)


def summarize(source, results):
    assert len(source) == len(results)
    counts = Counter()
    for row, result in zip(source, results):
        assert (row['session'], row['id']) == (result['session'], result['id'])
        answer = result['ans']; pred = answer['choice']
        counts['rows'] += 1
        is_name = row['gold'] != 'LEAVE'
        counts['names'] += is_name
        counts['candidate_hits'] += is_name and row['gold_in_opts']
        counts['correct_decisions'] += pred == (row['gold'] if row['gold_in_opts'] else 'OTHER')
        counts['correct_spellings'] += is_name and pred == row['gold']
        counts['wrong_names'] += pred not in ('OTHER', 'LEAVE', row['gold'])
        counts['OTHER'] += pred == 'OTHER'
        counts['LEAVE'] += pred == 'LEAVE'
        if pred not in ('OTHER', 'LEAVE') and answer['confidence'] >= .9:
            counts['confident_name_suggestions'] += 1
            counts['confident_correct_spellings'] += pred == row['gold']
    return dict(counts)


def score(args):
    manifest = read(args.workdir / 'MANIFEST.json')
    assert digest(args.data / 'e2_pick.json') == manifest['source_sha256']
    assert digest(args.data / 'results/e2/gemini.json') == manifest['baseline_result_sha256']
    source = read(args.data / 'e2_pick.json')
    baseline = read(args.data / 'results/e2/gemini.json')
    # Exact previously corrected tokens would normally be handled by the glossary
    # before model selection. Report the genuinely new-token subset separately.
    prior_exact = set()
    by_date = defaultdict(list)
    for row in source:
        by_date[row['session'][:10]].append(row)
    known = set()
    for date in sorted(by_date):
        for row in by_date[date]:
            if row['token'].casefold() in known:
                prior_exact.add((row['session'], row['id']))
        known.update(r['token'].casefold() for r in by_date[date] if r['gold'] != 'LEAVE')

    def novel_counts(rows, results):
        pairs = [(r, a) for r, a in zip(rows, results)
                 if r['gold'] != 'LEAVE' and (r['session'], r['id']) not in prior_exact]
        return {'new_token_names': len(pairs),
                'new_token_correct_spellings': sum(a['ans']['choice'] == r['gold'] for r, a in pairs)}

    report = {'manifest': manifest, 'baseline8': summarize(source, baseline), 'arms': {}}
    report['prior_exact_alias_rows'] = len(prior_exact)
    report['baseline8'].update(novel_counts(source, baseline))
    for arm in ARMS:
        path = args.workdir / f'{arm}.json'
        assert digest(path) == manifest['arms'][arm]['sha256']
        rows = read(path)
        results = [json.loads(line) for line in (args.workdir / f'{arm}-results.jsonl').read_text().splitlines()]
        summary = summarize(rows, results)
        summary.update(novel_counts(rows, results))
        wins = losses = rescued = lost = 0
        for original, row, old, new in zip(source, rows, baseline, results):
            old_ok = old['ans']['choice'] == original['gold']
            new_ok = new['ans']['choice'] == original['gold']
            if original['gold'] != 'LEAVE':
                wins += new_ok and not old_ok; losses += old_ok and not new_ok
                rescued += not original['gold_in_opts'] and new_ok
                lost += original['gold_in_opts'] and not row['gold_in_opts']
        summary.update(new_correct_spellings=wins, lost_correct_spellings=losses,
                       corrected_previously_missing_names=rescued, dropped_original_candidates=lost,
                       recorded_input_tokens=sum(r['usage']['input_tokens'] for r in results))
        report['arms'][arm] = summary
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'score'))
    parser.add_argument('--data', type=Path, default=Path.home() / 'data/decision-eval/v1')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=Path.home() / 'out-of-the-abyss/out-of-the-abyss')
    parser.add_argument('--phonetics', type=Path, default=Path.home() / '.claude/skills/vtt-spell-pass/dmetaphone.py')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--output', type=Path, default=Path('/tmp/gemini-retrieval-scores.json'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
