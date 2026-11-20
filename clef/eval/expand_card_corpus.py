"""Freeze additional GM spelling-card rulings without changing campaign files.

Reuses original v1 records for previously evaluated cards. Newer whole sessions
are reserved for validation; human rulings and notes never enter feature text.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value):
    return html.unescape(re.sub(r'<[^>]*>', '', str(value or ''))).strip()


def features(item):
    if 't' in item and 'y' in item:
        return {'format': 'original_shared_card', 'card_text': '\n\n'.join(
            f'{label}: {text(item.get(key))}' for label, key in (
                ('Card', 't'), ('Approve', 'y'), ('Reject', 'n'), ('Evidence', 'ev')))}
    # Explicit allow-list: exclude embedded decision, note and other outcome fields.
    selected = {key: item[key] for key in ('token', 'canonical', 'section', 'count',
        'context', 'sibling', 'sibling_verdict', 'confidence', 'reason',
        'evidence_note', 'edit_mode', 'reject_as', 'source_lines', 'source_cues',
        'occurrences') if key in item}
    return {'format': 'structured_queue_reconstruction',
            'card_text': json.dumps(selected, ensure_ascii=False, sort_keys=True)}


def summarize(rows):
    return {'rows': len(rows), 'sessions': len({r['session'] for r in rows}),
            'verdicts': dict(Counter(r['target']['verdict'] for r in rows)),
            'concrete_proposals': sum(r['concrete_proposal'] for r in rows),
            'concrete_proposal_verdicts': dict(Counter(r['target']['verdict'] for r in rows if r['concrete_proposal'])),
            'formats': dict(Counter(r['features']['format'] for r in rows))}


def concrete(mapping):
    if isinstance(mapping, list):
        pairs = [(r.get('was'), r.get('now')) for r in mapping if isinstance(r, dict)]
    elif isinstance(mapping, dict):
        first = lambda v: v[0] if isinstance(v, list) and v else v
        pairs = [(first(mapping.get('token')), first(mapping.get('canonical')))]
    else:
        pairs = []
    return any(isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()
               and a.casefold() != b.casefold() and b.casefold() not in ('none', '(none)') for a, b in pairs)


def build(campaign, data, output):
    if output.exists():
        raise ValueError('Choose a new output directory; an existing snapshot is never overwritten')
    frozen = {(r['session'], r['id']): r for r in read(data / 'e1_cards_prose.json')}
    raw = {(r['session'], r['id']): r for r in read(data / 'e1_cards_rawC.json')}
    last_date = max(session[:10] for session, _ in frozen)
    rows, sources, anomalies = [], {}, []
    seen = set()
    primary_files = []
    for session in sorted((campaign / 'summaries').iterdir()):
        for decision_path, item_path in (
            (session / 'spell_review/decisions.json', session / 'spell_review/review_items.json'),
            (session / 'vtt_spell_pass_decisions.json', session / 'vtt_spell_pass_review.json')):
            if not decision_path.exists():
                continue
            primary_files.append(decision_path)
            decisions, queue = read(decision_path), read(item_path)
            items = {str(it['id']): it for it in queue['items']}
            map_path = decision_path.with_name('review_map.json')
            mapping = read(map_path) if map_path.exists() else {}
            if map_path.exists():
                sources[str(map_path.relative_to(campaign))] = sha(map_path)
            qid, did = queue.get('reviewId', queue.get('review_id')), decisions.get('reviewId', decisions.get('review_id'))
            mismatch = bool(qid and did and qid != did)
            if mismatch:
                anomalies.append({'kind': 'review_id_mismatch', 'session': session.name,
                                  'preserved_from_v1': all((session.name, key) in frozen for key in decisions['decisions'])})
            for path in (decision_path, item_path):
                sources[str(path.relative_to(campaign))] = sha(path)
            for key, verdict in decisions['decisions'].items():
                identity = session.name, key
                if identity in seen:
                    raise ValueError(f'Duplicate primary ruling: {identity}')
                seen.add(identity)
                if key not in items:
                    raise ValueError(f'Decision has no matching card: {identity}')
                if verdict not in ('approve', 'reject', 'discuss', 'chatter'):
                    raise ValueError(f'Unknown verdict: {identity}, {verdict}')
                old = frozen.get(identity)
                if old:
                    if old['verdict'] != verdict:
                        raise ValueError(f'Changed historical label requires adjudication: {identity}')
                    view = {'format': 'frozen_v1', 'card_text': old['state'], 'raw_state': raw[identity]['state']}
                    fields = {}
                    for label, field in (('Transcript token', 'token'), ('Proposed replacement', 'canonical')):
                        match = re.search(r'^' + label + r': (.*)$', raw[identity]['state'], re.M)
                        fields[field] = match[1].strip() if match else ''
                    proposal = concrete(fields)
                else:
                    if mismatch:
                        raise ValueError(f'New ruling has conflicting review IDs: {identity}')
                    view = features(items[key])
                    proposal = concrete(mapping.get(key, items[key]))
                split = 'holdout' if not old and session.name[:10] > last_date else 'development'
                note = decisions.get('notes', {}).get(key, '')
                uid = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                rows.append({'record_id': uid, 'session': session.name, 'id': key, 'split': split,
                             'in_original_v1': bool(old), 'features': view,
                             'concrete_proposal': proposal,
                             'target': {'verdict': verdict, 'approve_as_presented': verdict == 'approve',
                                        'human_note': note},
                             'source': {'decisions': str(decision_path.relative_to(campaign)),
                                        'items': str(item_path.relative_to(campaign)),
                                        'saved_at': decisions.get('savedAt')}})
    if set(frozen) - seen:
        raise ValueError('Some original frozen records no longer have a primary source')
    # Report exact feature duplicates that would cross development and holdout.
    feature_hash = lambda r: hashlib.sha256(json.dumps(r['features'], sort_keys=True).encode()).hexdigest()
    dev_hashes = {feature_hash(r) for r in rows if r['split'] == 'development'}
    for row in rows:
        if row['split'] == 'holdout' and feature_hash(row) in dev_hashes:
            row['split'] = 'duplicate_quarantine'
            anomalies.append({'kind': 'cross_split_duplicate', 'record_id': row['record_id']})
    backups = []
    for path in sorted((campaign / 'summaries').glob('*/spell_review/decisions_saved_*.json')):
        backup, current = read(path), read(path.with_name('decisions.json'))
        backups.append({'path': str(path.relative_to(campaign)), 'sha256': sha(path),
                        'rows': len(backup.get('decisions', {})),
                        'same_labels_as_primary': backup.get('decisions') == current.get('decisions')})
    new = [r for r in rows if not r['in_original_v1']]
    report = {'all': summarize(rows), 'original_v1': summarize([r for r in rows if r['in_original_v1']]),
              'additional': summarize(new),
              'splits': {s: summarize([r for r in rows if r['split'] == s])
                         for s in sorted({r['split'] for r in rows})},
              'primary_decision_files': len(primary_files), 'backup_exports_not_counted': backups,
              'anomalies': anomalies,
              'original_v1_inputs': {name: sha(data / name) for name in ('e1_cards_prose.json', 'e1_cards_rawC.json')},
              'source_hashes': sources,
              'policy': {'target': 'approve as presented versus needs an individual ruling (reject/discuss/chatter)',
                         'holdout': f'Whole previously untested sessions dated after {last_date}; never randomly split cards from one session.',
                         'features': 'Explicit allow-list; human verdicts and notes are targets only. Old examples retain the original frozen text.',
                         'training_status': 'Dataset preparation only; no model fitted and no new data sent to a hosted API.'}}
    output.mkdir(parents=True)
    for split in report['splits']:
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                                   for r in rows if r['split'] == split))
    (output / 'MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=Path.home() / 'out-of-the-abyss/out-of-the-abyss')
    parser.add_argument('--data', type=Path, default=Path.home() / 'data/decision-eval/v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = build(args.campaign, args.data, args.output)
    # Aggregate report omits per-file paths and card text, but preserves audit issues.
    aggregate = {k: v for k, v in report.items() if k != 'source_hashes'}
    if args.report:
        args.report.write_text(json.dumps(aggregate, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('all', 'original_v1', 'additional', 'splits')}, indent=2))


if __name__ == '__main__':
    main()
