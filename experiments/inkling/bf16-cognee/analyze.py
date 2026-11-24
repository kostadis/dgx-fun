"""Audit a source-reviewed 182K BF16 Cognee replay without exporting passages."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('paired_native_audit', REPO / 'experiments/inkling/native-thinking/analyze.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); trial = root.parent
    source = trial / 'adaptive-context-8'; previous = trial / 'inkling-native-thinking-paired-20261122/off'
    control = A.load(root / 'control.json'); assert control['complete'] and control['services_unchanged']
    assert control['questions_returncode'] == control['agent_returncode'] == 0
    protected = A.load(root / 'protected-inputs.json')
    assert all(A.sha(trial / name) == digest for name, digest in protected.items())
    before = A.load(root / 'services-before.json')
    for name in ['services-after-questions.json', 'services-after-agent.json', 'services-after.json']:
        after = A.load(root / name)
        assert after['native'] == before['native'] and after['clef'] == before['clef']
        assert after['guard_active'] and not after['guard_stats']['floor_tripped']
        assert after['guard_stats']['container_id'] == before['native']['id']
        assert after['clef_health']['ready'] and after['available_bytes'] >= 12 * 1024**3
        assert next(m for m in after['models']['data'] if m['id'] == 'inkling-2-mini')['max_model_len'] == 182144
    switched = A.load(root / 'switch-before.json')
    restored = A.load(root / 'restoration.json'); final = A.load(root / 'services-final-review.json')
    assert control['capacity_returncode'] == control['restoration_returncode'] == 0
    assert restored['passed'] and restored['guard_active']
    assert restored['clef'] == before['clef'] == switched['clef'] == final['clef']
    assert restored['bf16']['running'] is False and final['bf16']['running'] is False
    assert restored['bf16']['id'] == final['bf16']['id'] == before['native']['id']
    assert final['fp8'] == restored['fp8'] and final['guard_active']
    assert final['guard_stats']['container_id'] == restored['fp8']['id']
    assert final['model']['max_model_len'] == 411520
    assert final['available_bytes'] >= 12 * 1024**3
    for key in ['id', 'image', 'command', 'name']:
        assert restored['fp8'][key] == switched['fp8'][key]
    assert restored['ready']['model']['max_model_len'] == 411520
    assert restored['smoke']['content'].strip() == '323'
    assert restored['smoke']['usage']['completion_tokens_details']['reasoning_tokens'] == 0
    validation = A.load(root / 'capacity-validation.json'); assert validation['passed']
    probe = validation['probe']
    assert validation['native'] == before['native'] and validation['clef'] == before['clef']
    assert probe['passed'] and probe['advertised_context'] == 182144
    assert 181044 <= probe['tokenized_prompt_tokens'] <= 181120
    assert probe['usage']['prompt_tokens'] == probe['tokenized_prompt_tokens']
    assert json.loads(probe['content']) == probe['expected'] and probe['finish_reason'] == 'stop'
    assert probe['reasoning_characters'] == 0 and probe['usage']['completion_tokens_details']['reasoning_tokens'] == 0
    assert not validation['memory']['floor_tripped']
    operation_hashes = A.load(root / 'operation-source-sha256.json')
    for name,path in [('operate.py','experiments/inkling/bf16-cognee/operate.py'),
                      ('capacity.py','experiments/inkling/context/run.py'),('monitor.py','experiments/inkling/context/monitor.py')]:
        assert A.sha(REPO / path) == operation_hashes[name]
    native = before['native']; command = native['command']
    normalize_command = lambda args: [v if i != args.index('--kv-cache-dtype') + 1 else 'CACHE_DTYPE'
                                      for i, v in enumerate(args)]
    assert normalize_command(command) == normalize_command(switched['fp8']['command'])
    assert command[command.index('--kv-cache-dtype') + 1] == 'auto'
    assert command[command.index('--kv-cache-memory-bytes') + 1] == '13958643712'
    assert native == A.load(root / 'native-startup.json')['native']
    code = A.load(root / 'source-sha256.json')
    assert all(A.sha(REPO / name) == digest for name, digest in code.items())
    expected_graph = A.load(source / 'progress.json')['graph']
    manifest = A.load(source / 'manifest.json')
    chunks = {c['id']: {**c, 'doc_id': d['doc_id']} for f in (source / 'chunks').glob('*.json')
              for d in [A.load(f)] for c in d['chunks']}
    cases = A.load(trial / 'frozen/cases.json')
    qroot = root / 'questions'; qp = A.load(qroot / 'protocol.json'); qs = A.load(qroot / 'progress.json')
    qr = A.calls(qroot); recalls = [A.load(f) for f in sorted((qroot / 'recall').glob('*.json'))]
    assert qs['passed'] and qs['graph_counts'] == expected_graph
    assert len(qr) == len(recalls) == len(cases) == 48
    assert qp['max_output_tokens'] == 4096 and qp['continue_on_truncation']
    assert qp['cases_sha256'] == A.sha(trial / 'frozen/cases.json')
    assert qp['source_manifest_sha256'] == A.sha(source / 'manifest.json')
    assert qp['script_sha256'] == A.load(previous / 'questions/protocol.json')['script_sha256']
    reviews = A.load(root / 'factual-review.json'); assert set(reviews) == {c['id'] for c in cases}
    byid = {c['id']: c for c in qr}; per_question = []
    assert len({r['session_id'] for r in recalls}) == 48
    for recall, case in zip(recalls, cases):
        assert recall['id'] == case['id'] and len(recall['llm_calls']) == 1
        c = byid[recall['llm_calls'][0]['id']]; b = c['forwarded']
        assert b['max_tokens'] == 4096 and b['messages'][0]['content'] == qp['system_prompt']
        assert 'response_format' not in b
        prompt = '\n'.join(m['content'] for m in b['messages'])
        assert case['question'] in prompt
        assert not any(other['question'] in prompt for other in cases if other['id'] != case['id'])
        old = A.load(previous / 'questions/recall' / (case['id'] + '.json'))
        old_call = A.load(previous / 'questions/calls' / (old['llm_calls'][0]['id'] + '.json'))
        normalize = lambda body: {k: v for k, v in body.items() if k != 'cache_salt'}
        assert normalize(b) == normalize(old_call['forwarded'])
        assert c['response']['usage']['prompt_tokens'] == old_call['response']['usage']['prompt_tokens']
        assert c['response']['choices'][0]['message']['content'] == old_call['response']['choices'][0]['message']['content']
        label = reviews[case['id']]
        assert label['correctness'] in ['correct_and_complete', 'partial', 'incorrect', 'abstained', 'source_ambiguous']
        assert label['grounding'] in ['supported', 'unsupported', 'unassessed']
        assert (label['correctness'] == 'source_ambiguous') == (case['id'] == 'q37')
        if not recall['successful_reader']:
            assert label['correctness'] != 'correct_and_complete'
        per_question.append({'id': case['id'], 'seconds': recall['seconds'],
            'successful_reader': recall['successful_reader'], 'quality': {k: label[k] for k in ['correctness', 'grounding']},
            'usage': c['response']['usage'], 'upstream_seconds': c['upstream_seconds'],
            'reader_request_matches_bf16_off_except_cache_salt': True})
    embeddings = [A.load(f) for f in (qroot / 'embeddings').glob('*.json')]
    assert embeddings and all(e['status'] == 200 and e['truncate'] is False
        and e['model'] == manifest['embedding_model'] and e['upstream'] == manifest['embedding_endpoint']
        and e['dimensions'] == 1024 for e in embeddings)
    latencies = [r['seconds'] for r in recalls]
    factual = {**A.summarize_calls(qr, 'off', qp['generation_extra_body']['cache_salt']),
        'questions': 48, 'successful_answers': sum(r['successful_reader'] for r in recalls),
        'query_sum_seconds': sum(latencies), 'p50_seconds': statistics.median(latencies),
        'p95_seconds': A.chunk_analysis.percentile(latencies, .95),
        'quality': {'adjudicable': 47, 'excluded': ['q37'],
            'correct_and_complete': sum(v['correctness'] == 'correct_and_complete' for v in reviews.values()),
            'grounded_complete': sum(v['correctness'] == 'correct_and_complete' and v['grounding'] == 'supported' for v in reviews.values()),
            'counts': dict(Counter(v['correctness'] for v in reviews.values())),
            'review_methods': dict(Counter(v.get('review_method', 'Codex source review') for v in reviews.values()))},
        'audit': A.verify_recall_context(list(chunks.values()), qr, recalls),
        'answers_and_reader_messages_exactly_matching_64k_bf16_off': 48, 'per_question': per_question}

    ar = root / 'agent'; ap = A.load(ar / 'protocol.json'); ast = A.load(ar / 'progress.json')
    rows = A.calls(ar); old = A.load(previous / 'agent/protocol.json')
    agent_embeddings = [A.load(f) for f in (ar / 'embeddings').glob('*.json')]
    assert agent_embeddings and all(e['status'] == 200 and e['truncate'] is False
        and e['model'] == manifest['embedding_model'] and e['upstream'] == manifest['embedding_endpoint']
        and e['dimensions'] == 1024 for e in agent_embeddings)
    assert ast['passed'] and ap['policy'] == old['policy'] and ap['action_schema'] == old['action_schema']
    assert ap['question_sha256'] == old['question_sha256'] and ap['top_k'] == 15 and ap['max_searches'] == 8
    assert ap['script_sha256'] == old['script_sha256'] and ap['reader_max_output_tokens'] == 4096
    first = next(c for c in rows if c['phase'] == 'search-01')
    old_first = next(c for c in A.calls(previous / 'agent') if c['phase'] == 'search-01')
    normalize = lambda body: {k: v for k, v in body.items() if k != 'cache_salt'}
    assert normalize(first['forwarded']) == normalize(old_first['forwarded'])
    rr = A.load(ar / 'result.json')
    old_result = A.load(previous / 'agent/result.json')
    old_rows = A.calls(previous / 'agent')
    assert len(rows) == len(old_rows)
    for current, prior in zip(rows, old_rows):
        assert normalize(current['forwarded']) == normalize(prior['forwarded'])
        assert current['response']['choices'][0]['message'].get('content') == prior['response']['choices'][0]['message'].get('content')
    assert rr['source_labels'] == old_result['source_labels']
    assert (ar / 'answer.txt').read_text() == (previous / 'agent/answer.txt').read_text()
    assert rr['graph_counts'] == expected_graph and not rr['seeded_from_prior_probes'] and not rr['manual_intervention_during_run']
    assert len({o['session_id'] for o in rr['observations']}) == len(rr['observations'])
    checks = 0; labels = {}; byid = {c['id']: c for c in rows}
    for index, o in enumerate(rr['observations']):
        if index:
            assert rr['decisions'][index - 1]['action']['query'] == o['query']
        for identity in sorted(o['retrieved_chunk_ids'], key=lambda i: (chunks[i]['doc_id'], chunks[i]['index'])):
            if identity not in labels:
                labels[identity] = f'S{len(labels) + 1:03d}'
        union = '\n\n'.join(f'[{label}] DOCUMENT {chunks[i]["doc_id"]} CHUNK {i}\n' + chunks[i]['text'] for i, label in labels.items())
        for c in rows:
            if c['phase'].startswith(f'decision-{o["step"]:02d}-'):
                u = c['forwarded']['messages'][1]['content']; assert u.endswith(union)
                history = json.loads(u.split('OBSERVATIONS IN ORDER (tool answers are unverified):\n', 1)[1].split('\nACCUMULATED RAW SOURCE PASSAGES:', 1)[0])
                assert len(history) == o['step'] and history[-1]['unverified_tool_answer'] == A.text(o['answers'])
                assert c['started_unix'] >= max(byid[k['id']]['started_unix'] + byid[k['id']]['seconds'] for k in o['llm_calls'])
                checks += 1
    assert labels == rr['source_labels']
    for c in rows:
        assert c['forwarded']['max_tokens'] == (4096 if c['phase'].startswith('search-') else 2800)
        if c['phase'] == 'budget-answer':
            assert c['forwarded']['messages'][1]['content'].endswith(union)
    cited = set(re.findall(r'\bS\d{3}\b', (ar / 'answer.txt').read_text()))
    assert cited and cited.issubset(set(labels.values()))
    agent = {**A.summarize_calls(rows, 'off', ap['generation_overrides']['cache_salt']),
        'completed': True, 'client_seconds': ast['finished_unix'] - ast['started_unix'],
        'research_seconds': rr['seconds'], 'searches': len(rr['observations']), 'raw_sources': len(labels),
        'protocol_corrections': len(list((ar / 'steps').glob('*invalid-action*.json'))),
        'outcome': rr['stop_reason'], 'initial_reader_request_matches_bf16_off_except_cache_salt': True,
        'full_trace_matches_64k_bf16_off_except_cache_salt': True,
        'final_answer_and_source_union_match_64k_bf16_off': True,
        'audit': {**A.verify_recall_context(list(chunks.values()), rows, rr['observations']),
                  'latest_observation_and_union_checks': checks},
        'source_review': A.load(root / 'agent-review-public.json')}
    export = {'experiment': 'Cognee replay on native 182K BF16 Inkling with generated thinking blocked',
        'complete': True, 'context_tokens': 182144, 'kv_cache_dtype': 'bfloat16', 'kv_budget_bytes': 13958643712,
        'started_unix': control['started_unix'], 'finished_unix': control['finished_unix'],
        'verified_utc': final['checked_utc'],
        'native': native, 'same_bf16_process_during_tests': True, 'clef_unchanged': True,
        'ingestion_model': manifest['model'], 'embedding': {k: manifest[k] for k in
            ['embedding_model', 'embedding_endpoint', 'embedding_digest']},
        'graph_counts': expected_graph, 'source_manifest_sha256': A.sha(source / 'manifest.json'),
        'source_sha256': code, 'operation_source_sha256': operation_hashes, 'protected_files_hash_verified': len(protected),
        'cache_salts': {k: A.load(root / (k + '-options.json'))['cache_salt'] for k in ['questions', 'agent']},
        'factual': factual, 'agent': agent,
        'operations': {'before_available_bytes': before['available_bytes'],
            'after_available_bytes': after['available_bytes'], 'guard_floor_tripped': False,
            'final_restored_available_bytes': final['available_bytes'],
            'guard_min_since_bf16_start_bytes': validation['memory']['min_available_bytes'],
            'bf16_process_unchanged_during_measurements': True, 'fp8_restored': True,
            'restored_fp8': restored['fp8'], 'qwen_model_ids': final['qwen_model_ids']},
        'capacity_validation': {key: probe[key] for key in ['advertised_context','tokenized_prompt_tokens',
            'output_cap','prompt_sha256','sentinel_placement','passed','seconds','usage','reasoning_characters','finish_reason']},
        'limits': ['One observation per factual question and one agent run; comparisons ran at other times.',
                   'The 182K and old 64K native runs retain BF16 KV but differ in context and cache planning; the 411K run uses FP8 KV.',
                   'The standard factual/agent prompts do not fill the 182K window; its capacity test is separate.',
                   'Agent queries and search work are chosen independently; elapsed time needs quality context.',
                   'Codex source review, not blinded human adjudication.']}
    a.output.write_text(json.dumps(export, indent=2) + '\n')
    print(json.dumps({k: {field: value for field, value in export[k].items() if field not in ['per_question', 'source_review']}
                      for k in ['factual', 'agent']}, indent=2))


if __name__ == '__main__':
    main()
