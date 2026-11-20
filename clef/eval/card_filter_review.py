"""Build a historical replay of likely-yes bulk approval plus individual cards.

No API calls or campaign edits. Scores come from the frozen E1 Gemini run.
Outputs contain private transcript excerpts; keep --output outside the repo.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re


def field(state, label):
    match = re.search(r'^' + re.escape(label) + r': (.*)$', state, re.M)
    return match[1].strip() if match else ''


def concrete_proposal(token, replacement):
    return bool(token and replacement and replacement.casefold() not in (
        token.casefold(), '(none)', 'none'))


def tier_for(proposal, p, high=.9, middle=.8):
    if not proposal or isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        return 'cards'
    return 'high' if p > high else 'middle' if p >= middle else 'cards'


def replay(data, threshold=.9, session=None, middle_threshold=.8, ablation_dir=None, model='gemini', scores_file=None):
    read = lambda name: json.loads((data / name).read_text())
    prose = {(r['session'], r['id']): r for r in read('e1_cards_prose.json')}
    raw = {(r['session'], r['id']): r for r in read('e1_cards_rawC.json')}
    model_id = {'gemini': 'gemini-decision-1', 'clef': 'clef', 'clef-flash': 'clef-flash'}[model]
    scores = json.loads(scores_file.read_text()) if scores_file else read(f'results/e1_rawC/{model}.json')
    overrides = {}
    if ablation_dir:
        report = json.loads((ablation_dir / 'summary.json').read_text())
        if report['model'] != model_id:
            raise ValueError('Corrected-input scores belong to a different model')
        prompts = json.loads((ablation_dir / 'prompts.json').read_text())
        state = prompts['states']['faithful_edit']
        if hashlib.sha256(state.encode()).hexdigest() != report['variants']['faithful_edit']['input_sha256']:
            raise ValueError('Corrected-input score does not match its prompt')
        overrides[('2026-01-12-chapter_43-harvesting-horror-in-the-underdark', 'c_entemok_moon')] = (
            report['variants']['faithful_edit']['median'], state)
    items, aggregate = [], Counter(dict.fromkeys((
        'items', 'flat_list', 'middle_list', 'individual_cards', 'proposed_corrections',
        'historical_nonapprovals_in_flat_list'), 0))
    tier_counts, tier_nonapprovals = Counter(dict.fromkeys(('high', 'middle', 'cards'), 0)), Counter(dict.fromkeys(('high', 'middle', 'cards'), 0))
    for scored in scores:
        key = scored['session'], scored['id']
        if session and key[0] != session:
            continue
        source, card = raw[key]['state'], prose[key]['state']
        token, replacement = field(source, 'Transcript token'), field(source, 'Proposed replacement')
        p = scored.get('answers', {}).get('accept', {}).get('noul')
        updated = key in overrides
        if updated:
            p, source = overrides[key]
        proposal = concrete_proposal(token, replacement)
        tier = tier_for(proposal, p, threshold, middle_threshold)
        likely = tier == 'high'
        uid = 'r-' + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        title = field(card, 'Card') or token or 'Spelling question'
        proposed_text = card.partition('\n\nProposal: ')[2].partition('\n\nEvidence: ')[0]
        item = {'id': uid, 'title': title, 'token': token, 'replacement': replacement,
                'session': key[0], 'original_id': key[1], 'p_yes': p,
                'likely_yes': likely, 'tier': tier, 'updated_score': updated,
                'display_pair': title.split(' ·', 1)[0] if '→' in title else '',
                'proposal': card, 'proposed_text': proposed_text, 'evidence': source}
        items.append(item)
        tier_counts[tier] += 1
        if scored['verdict'] != 'approve': tier_nonapprovals[tier] += 1
        aggregate['items'] += 1
        aggregate[{'high': 'flat_list', 'middle': 'middle_list', 'cards': 'individual_cards'}[tier]] += 1
        if proposal:
            aggregate['proposed_corrections'] += 1
        if likely and scored['verdict'] != 'approve':
            aggregate['historical_nonapprovals_in_flat_list'] += 1
    if not items:
        raise ValueError('No matching review items')
    # Historical verdicts and notes are deliberately absent from browser data.
    signature = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:20]
    spec = {'title': 'Spelling review · replay', 'reviewId': f'card-filter-replay:{signature}',
            'model': model_id, 'threshold': threshold, 'middle_threshold': middle_threshold, 'items': items}
    return spec, {**dict(aggregate), 'tier_counts': dict(tier_counts),
                  'historical_nonapprovals_by_tier': dict(tier_nonapprovals),
                  'corrected_input_scores_used': sum(it['updated_score'] for it in items)}


def render(spec):
    template = Path(__file__).with_suffix('.html').read_text()
    # Transcript text is JSON, never executable HTML, including closing script tags.
    payload = json.dumps(spec, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return template.replace('__REVIEW_DATA__', payload)


def shared_spec(spec):
    return {'title': spec['title'], 'reviewId': spec['reviewId'], 'items': [
        {'id': item['id'], 't': html.escape(item['title']),
         'y': 'Record approval of this historical proposed correction in the replay only.',
         'n': 'Record rejection in the replay only; no campaign files change.',
         'ev': html.escape(item['evidence'])} for item in spec['items']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.home() / 'data/decision-eval/v1')
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--middle-threshold', type=float, default=.8)
    parser.add_argument('--ablation-dir', type=Path)
    parser.add_argument('--model', choices=('gemini', 'clef', 'clef-flash'), default='gemini')
    parser.add_argument('--scores-file', type=Path)
    parser.add_argument('--session')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.middle_threshold < args.threshold <= 1:
        parser.error('thresholds must satisfy 0 <= middle < high <= 1')
    spec, counts = replay(args.data, args.threshold, args.session, args.middle_threshold,
                          args.ablation_dir, args.model, args.scores_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(spec))
    args.output.with_suffix('.items.json').write_text(json.dumps(shared_spec(spec), indent=2))
    args.output.with_suffix('.audit.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
