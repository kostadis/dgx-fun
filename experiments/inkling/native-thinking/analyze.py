"""Audit a paired native thinking replay and export source-reviewed metrics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'experiments/adaptive-chunking/cognee'))
import analyze as chunk_analysis
from audit_context import verify_recall_context


def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def calls(root): return sorted((load(f) for f in (root / 'calls').glob('*.json')), key=lambda c: c['started_unix'])


def text(answers):
    return '\n'.join(re.split(r'\n\s*Evidence:', a['text'], maxsplit=1, flags=re.I)[0] for a in answers)


def summarize_calls(rows, arm, salt):
    usage = Counter(); reasoning = chars = cached = cached_reports = 0
    for c in rows:
        b = c['forwarded']; response = c['response']; u = response['usage']
        assert c['status'] == 200 and b['model'] == 'inkling-2-mini'
        assert b['reasoning_effort'] == 'none' and b['chat_template_kwargs']['enable_thinking'] is False
        assert b['temperature'] == 0 and b['seed'] == 604 and b['cache_salt'] == salt
        assert b.get('bad_words', []) == (['<|content_thinking|>'] if arm == 'off' else [])
        assert all(v['finish_reason'] in ('stop', 'length') for v in response['choices'])
        usage.update({k: u[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        reasoning += (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        chars += sum(len(v['message'].get('reasoning_content') or v['message'].get('reasoning') or '') for v in response['choices'])
        cached += (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        cached_reports += 'cached_tokens' in (u.get('prompt_tokens_details') or {})
    if arm == 'off': assert reasoning == chars == 0
    return {'llm_calls': len(rows), 'usage': dict(usage), 'reasoning_tokens': reasoning,
            'reasoning_characters': chars, 'cached_prompt_tokens': cached if cached_reports == len(rows) else None,
            'calls_reporting_cached_prompt_tokens': cached_reports,
            'upstream_seconds': sum(c['upstream_seconds'] for c in rows),
            'finish_reasons': dict(Counter(v['finish_reason'] for c in rows for v in c['response']['choices']))}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    root = a.root.resolve(); trial = root.parent; source = trial / 'adaptive-context-8'
    control = load(root / 'control.json'); assert control['complete']
    protected = load(root / 'protected-inputs.json')
    assert all(sha(trial / name) == value for name, value in protected.items())
    before = load(root / 'services-before.json'); startup = load(root / 'native-startup.json')
    restore = load(root / 'restoration.json'); assert restore['passed']
    assert restore['clef'] == before['clef']
    assert restore['native']['id'] == startup['native']['id'] == before['native']['id']
    assert restore['native']['image'] == startup['native']['image'] == before['native']['image']
    assert restore['native']['started'] == startup['native']['started']
    assert restore['native']['command'] == before['native']['command']
    assert restore['dspark_start']['command'] == before['dspark_start']['command']
    cases = load(trial / 'frozen/cases.json'); expected = load(source / 'progress.json')['graph']
    chunks = {c['id']: {**c, 'doc_id': d['doc_id']} for f in (source / 'chunks').glob('*.json')
              for d in [load(f)] for c in d['chunks']}
    result = {}; all_question_calls = {}
    for arm in ('on', 'off'):
        base = root / arm; qroot = base / 'questions'; qp = load(qroot / 'protocol.json')
        qs = load(qroot / 'progress.json'); qr = calls(qroot)
        recalls = [load(f) for f in sorted((qroot / 'recall').glob('*.json'))]
        assert qs['passed'] and qs['graph_counts'] == expected
        assert len(qr) == len(recalls) == len(cases) == 48
        assert len({r['session_id'] for r in recalls}) == 48
        assert qp['cases_sha256'] == sha(trial / 'frozen/cases.json')
        assert qp['source_manifest_sha256'] == sha(source / 'manifest.json')
        assert qp['max_output_tokens'] == 4096 and qp['continue_on_truncation']
        reviews = load(base / 'factual-review.json'); assert set(reviews) == {c['id'] for c in cases}
        byid = {c['id']: c for c in qr}; per_question = []; indexed = {}
        for recall, case in zip(recalls, cases):
            assert recall['id'] == case['id'] and len(recall['llm_calls']) == 1
            c = byid[recall['llm_calls'][0]['id']]; b = c['forwarded']; indexed[case['id']] = c
            assert b['max_tokens'] == 4096 and b['messages'][0]['content'] == qp['system_prompt']
            assert 'response_format' not in b
            prompt = '\n'.join(m['content'] for m in b['messages'])
            assert case['question'] in prompt
            assert not any(o['question'] in prompt for o in cases if o['id'] != case['id'])
            label = reviews[case['id']]
            assert label['correctness'] in ('correct_and_complete', 'partial', 'incorrect', 'abstained', 'source_ambiguous')
            assert label['grounding'] in ('supported', 'unsupported', 'unassessed')
            assert (label['correctness'] == 'source_ambiguous') == (case['id'] == 'q37')
            if not recall['successful_reader']: assert label['correctness'] != 'correct_and_complete'
            per_question.append({'id': case['id'], 'seconds': recall['seconds'],
                                 'successful_reader': recall['successful_reader'],
                                 'quality': {k: label[k] for k in ('correctness', 'grounding')},
                                 'usage': c['response']['usage'], 'upstream_seconds': c['upstream_seconds']})
        all_question_calls[arm] = indexed
        embeddings = [load(f) for f in (qroot / 'embeddings').glob('*.json')]
        assert embeddings and all(e['status'] == 200 and e['truncate'] is False for e in embeddings)
        latencies = [r['seconds'] for r in recalls]
        factual = {**summarize_calls(qr, arm, qp['generation_extra_body']['cache_salt']),
                   'reader_runner_sha256': qp['script_sha256'], 'capture_sha256': qp['capture_sha256'],
                   'questions': 48, 'successful_answers': sum(r['successful_reader'] for r in recalls),
                   'query_sum_seconds': sum(latencies), 'p50_seconds': statistics.median(latencies),
                   'p95_seconds': chunk_analysis.percentile(latencies, .95),
                   'quality': {'adjudicable': 47, 'excluded': ['q37'],
                               'correct_and_complete': sum(v['correctness'] == 'correct_and_complete' for v in reviews.values()),
                               'grounded_complete': sum(v['correctness'] == 'correct_and_complete' and v['grounding'] == 'supported' for v in reviews.values()),
                               'counts': dict(Counter(v['correctness'] for v in reviews.values())),
                               'review_methods': dict(Counter(v.get('review_method', 'Codex source review') for v in reviews.values()))},
                   'audit': verify_recall_context(list(chunks.values()), qr, recalls), 'per_question': per_question}

        ar = base / 'agent'; ap = load(ar / 'protocol.json'); ast = load(ar / 'progress.json'); rows = calls(ar)
        old = load(trial / 'inkling-schema-sequential-20261121/adaptive/protocol.json')
        assert ap['policy'] == old['policy'] and ap['action_schema'] == old['action_schema']
        assert ap['question_sha256'] == old['question_sha256'] and ap['top_k'] == 15 and ap['max_searches'] == 8
        agent = {**summarize_calls(rows, arm, ap['generation_overrides']['cache_salt']),
                 'agent_runner_sha256': ap['script_sha256'],
                 'completed': ast['passed'], 'client_seconds': ast['finished_unix'] - ast['started_unix'],
                 'protocol_corrections': len(list((ar / 'steps').glob('*invalid-action*.json'))),
                 'source_review': load(base / 'agent-review-public.json')}
        first = next(c for c in rows if c['phase'] == 'search-01')
        first_body = {k: v for k, v in first['forwarded'].items() if k not in ('cache_salt', 'bad_words')}
        agent['initial_reader_nonintervention_request_sha256'] = hashlib.sha256(json.dumps(first_body, sort_keys=True).encode()).hexdigest()
        if ast['passed']:
            rr = load(ar / 'result.json'); assert rr['graph_counts'] == expected
            assert not rr['seeded_from_prior_probes'] and not rr['manual_intervention_during_run']
            assert len({o['session_id'] for o in rr['observations']}) == len(rr['observations'])
            checks = 0; labels = {}; byid = {c['id']: c for c in rows}
            for index, o in enumerate(rr['observations']):
                if index: assert rr['decisions'][index - 1]['action']['query'] == o['query']
                for i in sorted(o['retrieved_chunk_ids'], key=lambda i: (chunks[i]['doc_id'], chunks[i]['index'])):
                    if i not in labels: labels[i] = f'S{len(labels) + 1:03d}'
                union = '\n\n'.join(f'[{label}] DOCUMENT {chunks[i]["doc_id"]} CHUNK {i}\n' + chunks[i]['text'] for i, label in labels.items())
                for c in rows:
                    if c['phase'].startswith(f'decision-{o["step"]:02d}-'):
                        u = c['forwarded']['messages'][1]['content']; assert u.endswith(union)
                        history = json.loads(u.split('OBSERVATIONS IN ORDER (tool answers are unverified):\n', 1)[1].split('\nACCUMULATED RAW SOURCE PASSAGES:', 1)[0])
                        assert len(history) == o['step'] and history[-1]['unverified_tool_answer'] == text(o['answers'])
                        assert c['started_unix'] >= max(byid[k['id']]['started_unix'] + byid[k['id']]['seconds'] for k in o['llm_calls'])
                        checks += 1
            assert labels == rr['source_labels']
            for c in rows:
                assert c['forwarded']['max_tokens'] == (4096 if c['phase'].startswith('search-') else 2800)
                if c['phase'] == 'budget-answer':
                    assert c['forwarded']['messages'][1]['content'].endswith(union)
            agent.update(searches=len(rr['observations']), raw_sources=len(labels), outcome=rr['stop_reason'],
                         research_seconds=rr['seconds'], audit={**verify_recall_context(list(chunks.values()), rows, rr['observations']),
                                                               'latest_observation_and_union_checks': checks})
        else:
            agent.update(outcome='failed', error_type=ast.get('error', '').split('(', 1)[0])
        result[arm] = {'factual': factual, 'agent': agent}

    assert result['on']['factual']['reader_runner_sha256'] == result['off']['factual']['reader_runner_sha256']
    assert result['on']['agent']['agent_runner_sha256'] == result['off']['agent']['agent_runner_sha256']
    assert result['on']['agent']['initial_reader_nonintervention_request_sha256'] == result['off']['agent']['initial_reader_nonintervention_request_sha256']
    paired = []
    for case in cases:
        on, off = (all_question_calls[k][case['id']] for k in ('on', 'off'))
        def normalized(c): return {k: v for k, v in c['forwarded'].items() if k not in ('cache_salt', 'bad_words')}
        assert normalized(on) == normalized(off)
        assert on['response']['usage']['prompt_tokens'] == off['response']['usage']['prompt_tokens']
        paired.append({'id': case['id'], 'same_reader_messages_and_nonintervention_parameters': True})
    on, off = (result[arm]['factual'] for arm in ('on', 'off'))
    comparison = {'factual_p50_speedup': on['p50_seconds'] / off['p50_seconds'],
                  'factual_query_sum_speedup': on['query_sum_seconds'] / off['query_sum_seconds'],
                  'factual_p50_reduction_fraction': 1 - off['p50_seconds'] / on['p50_seconds'],
                  'factual_query_sum_reduction_fraction': 1 - off['query_sum_seconds'] / on['query_sum_seconds'],
                  'agent_client_time_ratio': result['on']['agent']['client_seconds'] / result['off']['agent']['client_seconds'],
                  'agent_search_work_differs': True}
    output = {'experiment': 'Native Inkling minimum-effort thinking versus thinking-channel suppression',
              'complete': True, 'arms': result, 'paired_request_audit': paired,
              'comparison': comparison,
              'native_image': startup['native']['image'], 'model_revision': 'b85c3cfa615fc413cc9af3379f491b963f3b586a',
              'native_container': startup['native'], 'same_process_both_arms': True,
              'thinking_off_control': {'bad_words': ['<|content_thinking|>']}, 'requested_effort_both_arms': 'none',
              'cache_isolation': load(root / 'cache-isolation.json'), 'graph_counts': expected,
              'source_manifest_sha256': sha(source / 'manifest.json'),
              'operations': {'clef_unchanged': True, 'dspark_restored': True,
                             'controller_source': load(root / 'controller-source-audit.json'),
                             'protected_files_hash_verified': len(protected), 'startup_seconds': startup['startup_seconds'],
                             'native_memory': load(root / 'off/memory-end.json')},
              'limits': ['One observation per factual question and one agent run per arm, in fixed on-then-off order.',
                         'Clef shared the device; concurrent activity was not controlled.',
                         'Reader and agent output caps are matched across arms; factual truncations count as failures.',
                         'Agents choose different queries, so agent elapsed times need quality and search-work context.',
                         'Source review is Codex assessment, not blinded human adjudication.']}
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({arm: {'factual': {k: result[arm]['factual'][k] for k in ('query_sum_seconds', 'p50_seconds', 'p95_seconds', 'quality', 'reasoning_tokens')},
                           'agent': {k: v for k, v in result[arm]['agent'].items() if k not in ('usage', 'source_review', 'audit')}}
                      for arm in ('on', 'off')}, indent=2))


if __name__ == '__main__': main()
