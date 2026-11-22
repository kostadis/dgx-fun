"""Audit completed DSpark Cognee runs; export metrics and reviewed labels only."""
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
from analyze import coverage, percentile, source_chunks
from audit_context import verify_recall_context


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(root):
    return sorted((load(p) for p in (root / 'calls').glob('*.json')),
                  key=lambda c: c['started_unix'])


def answer_text(answers):
    return '\n'.join(re.split(r'\n\s*Evidence:', a['text'], maxsplit=1, flags=re.I)[0]
                     for a in answers)


def call_metrics(rows, allow_truncated=False):
    usage = Counter()
    proposed = accepted = reasoning_characters = 0
    prompt_ms = predicted_ms = 0
    for c in rows:
        body = c['forwarded']
        assert c['status'] == 200
        finish = [v['finish_reason'] for v in c['response']['choices']]
        assert finish == ['stop'] or (allow_truncated and finish == ['length'])
        assert body['model'] == 'inkling-2-mini' and 'reasoning_effort' not in body
        assert body['temperature'] == 0 and body['seed'] == 604
        assert body['chat_template_kwargs']['enable_thinking'] is False
        assert body['chat_template_kwargs']['reasoning_effort'] == 0.0
        assert body['logit_bias'] == {'200008': False}
        usage.update({k: c['response']['usage'][k]
                      for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        timings = c['response'].get('timings', {})
        assert timings.get('cache_n') == 0
        proposed += timings.get('draft_n', 0)
        accepted += timings.get('draft_n_accepted', 0)
        prompt_ms += timings.get('prompt_ms', 0)
        predicted_ms += timings.get('predicted_ms', 0)
        message = c['response']['choices'][0]['message']
        if finish == ['stop']:
            assert message.get('content')
        reasoning_characters += len(message.get('reasoning_content') or '')
        assert not message.get('reasoning_content')
        assert '<|content_thinking|>' not in (message.get('content') or '')
    return {'llm_calls': len(rows), 'usage': dict(usage),
            'upstream_seconds': sum(c['upstream_seconds'] for c in rows),
            'prompt_evaluation_seconds': prompt_ms / 1000,
            'generation_seconds': predicted_ms / 1000,
            'draft_tokens_proposed': proposed, 'draft_tokens_accepted': accepted,
            'draft_acceptance_rate': accepted / proposed if proposed else None,
            'finish_reasons': dict(Counter(v['finish_reason'] for c in rows for v in c['response']['choices'])),
            'reasoning_characters': reasoning_characters,
            'reasoning_token_count_available': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve(); trial = root.parent; source = trial / 'adaptive-context-8'
    control = load(root / 'control.json')
    assert control['complete']
    protected = load(root / 'protected-inputs.json')
    assert all(sha(trial / name) == digest for name, digest in protected.items())
    before, after = (load(root / ('services-' + phase + '.json')) for phase in ('before', 'after'))
    assert before['spark2']['clef'] == after['spark2']['clef']
    assert before['spark2']['active_pid'] == after['spark2']['active_pid']
    assert not after['spark2']['memory']['floor_tripped']
    manifest = load(source / 'manifest.json')
    counts = load(source / 'progress.json')['graph']
    chunks = {c['id']: {**c, 'doc_id': d['doc_id']}
              for f in (source / 'chunks').glob('*.json')
              for d in [load(f)] for c in d['chunks']}
    lineage = source_chunks(source, 'adaptive')
    cases = load(trial / 'frozen/cases.json')

    qroot = root / 'questions'; qstate = load(qroot / 'progress.json')
    qp = load(qroot / 'protocol.json'); qrows = calls(qroot)
    recalls = [load(f) for f in sorted((qroot / 'recall').glob('*.json'))]
    assert qstate['passed'] and qstate['graph_counts'] == counts
    assert len(recalls) == len(qrows) == len(cases) == 48
    assert [r['id'] for r in recalls] == [c['id'] for c in cases]
    assert len({r['session_id'] for r in recalls}) == 48
    assert qp['cases_sha256'] == sha(trial / 'frozen/cases.json')
    assert qp['source_manifest_sha256'] == sha(source / 'manifest.json')
    assert qp['max_output_tokens'] == 4096 and qp['continue_on_truncation']
    assert qp['generation_extra_body']['logit_bias'] == {'200008': False}
    byid = {c['id']: c for c in qrows}
    qaudit = verify_recall_context(list(chunks.values()), qrows, recalls)
    reviews = load(root / 'factual-review.json')
    assert set(reviews) == {c['id'] for c in cases}
    allowed_correctness = {'correct_and_complete', 'partial', 'incorrect', 'abstained', 'source_ambiguous'}
    allowed_grounding = {'supported', 'unsupported', 'unassessed'}
    metrics = []
    for recall, case in zip(recalls, cases):
        assert len(recall['llm_calls']) == 1
        row = byid[recall['llm_calls'][0]['id']]; body = row['forwarded']
        assert row['phase'] == 'recall-isolated-' + recall['id']
        assert body['messages'][0]['content'] == qp['system_prompt']
        assert body['max_tokens'] == 4096
        assert 'response_format' not in body
        prompt = '\n'.join(m['content'] for m in body['messages'])
        assert case['question'] in prompt
        assert not any(c['question'] in prompt for c in cases if c['id'] != recall['id'])
        baseline = load(source / 'recall' / (recall['id'] + '.json'))
        old_call = load(source / 'calls' / (baseline['llm_calls'][0]['id'] + '.json'))
        native_root = trial / 'inkling-questions-20261121/adaptive'
        native = load(native_root / 'recall' / (recall['id'] + '.json'))
        native_call = load(native_root / 'calls' / (native['llm_calls'][0]['id'] + '.json'))
        def raw_ids(r):
            return {e['artifact_id'] for v in r['answers'] for e in v['metadata']['evidence']
                    if e.get('kind') == 'graph_node' and e.get('role') == 'used_as_context'
                    and e['artifact_id'] in chunks}
        ids = raw_ids(recall)
        review = reviews[recall['id']]
        assert review['correctness'] in allowed_correctness and review['grounding'] in allowed_grounding
        assert (review['correctness'] == 'source_ambiguous') == (recall['id'] == 'q37')
        if not recall['successful_reader']:
            assert review['correctness'] != 'correct_and_complete'
        metrics.append({'id': recall['id'], 'seconds': recall['seconds'],
                        'qwen_seconds': baseline['seconds'], 'native_inkling_seconds': native['seconds'],
                        'reader_messages_identical_to_qwen': body['messages'] == old_call['forwarded']['messages'],
                        'reader_messages_identical_to_native_inkling': body['messages'] == native_call['forwarded']['messages'],
                        'raw_chunk_set_identical_to_qwen': ids == raw_ids(baseline),
                        'raw_evidence_coverage': coverage([lineage[i] for i in ids], case['evidence']),
                        'successful_reader': recall['successful_reader'],
                        'answer_words': len(answer_text(recall['answers']).split()),
                        'quality': {k: review[k] for k in ('correctness', 'grounding')},
                        'usage': row['response']['usage'], 'timings': row['response'].get('timings', {})})
    embeddings = [load(f) for f in (qroot / 'embeddings').glob('*.json')]
    assert embeddings and all(e['status'] == 200 and e['truncate'] is False for e in embeddings)
    times = [r['seconds'] for r in recalls]
    factual = {**call_metrics(qrows, allow_truncated=True), 'questions': 48,
               'successful_answers': sum(r['successful_reader'] for r in recalls),
               'truncated_questions': qstate.get('truncated_questions', []),
               'max_output_tokens': 4096, 'adjudicable_questions': 47, 'excluded': ['q37'],
               'recall_seconds': sum(times), 'recall_p50_seconds': statistics.median(times),
               'recall_p95_seconds': percentile(times, .95),
               'client_seconds_including_startup': qstate['finished_unix'] - qstate['started_unix'],
               'quality': {'correct_and_complete': sum(v['correctness'] == 'correct_and_complete' for v in reviews.values()),
                           'grounded_complete': sum(v['correctness'] == 'correct_and_complete' and v['grounding'] == 'supported'
                                                    for v in reviews.values()),
                           'counts': dict(Counter(v['correctness'] for v in reviews.values())),
                           'rubric': 'Frozen gold spans; q37 excluded as predeclared. Q38 scores trigger and round, without treating the worm as an ally.',
                           'q20_wording_sensitivity': {
                               'strict': reviews['q20']['correctness'],
                               'lenient_complete_if_read_as_intended': sum(v['correctness'] == 'correct_and_complete' for v in reviews.values()) + int(reviews['q20']['correctness'] != 'correct_and_complete'),
                               'same_sensitivity_as_native_inkling': True}},
               'audit': {**qaudit, 'unique_sessions': 48, 'graph_counts_unchanged': True,
                         **{k: sum(r[k] for r in metrics) for k in (
                             'reader_messages_identical_to_qwen', 'reader_messages_identical_to_native_inkling',
                             'raw_chunk_set_identical_to_qwen')}},
               'runner_sha256': qp['script_sha256'], 'cases_sha256': qp['cases_sha256'],
               'per_question': metrics}

    aroot = root / 'agent'; result = load(aroot / 'result.json')
    ap = load(aroot / 'protocol.json'); astate = load(aroot / 'progress.json'); arows = calls(aroot)
    assert astate['passed'] and result['graph_counts'] == counts
    assert not result['seeded_from_prior_probes'] and not result['manual_intervention_during_run']
    assert len({o['session_id'] for o in result['observations']}) == len(result['observations'])
    old_agent = trial / 'inkling-schema-sequential-20261121/adaptive'
    old_protocol = load(old_agent / 'protocol.json')
    assert ap['policy'] == old_protocol['policy']
    assert ap['action_schema'] == old_protocol['action_schema']
    assert ap['question_sha256'] == old_protocol['question_sha256']
    assert ap['top_k'] == old_protocol['top_k'] == 15
    assert ap['max_searches'] == old_protocol['max_searches'] == 8
    assert ap['reader_max_output_tokens'] == 4096
    aaudit = verify_recall_context(list(chunks.values()), arows, result['observations'])
    source_labels = {}; checks = 0; abyid = {c['id']: c for c in arows}
    for index, observation in enumerate(result['observations']):
        step = observation['step']
        if index:
            assert result['decisions'][index - 1]['action']['query'] == observation['query']
        for i in sorted(observation['retrieved_chunk_ids'], key=lambda i: (chunks[i]['doc_id'], chunks[i]['index'])):
            if i not in source_labels:
                source_labels[i] = f'S{len(source_labels) + 1:03d}'
        union = '\n\n'.join(f'[{label}] DOCUMENT {chunks[i]["doc_id"]} CHUNK {i}\n' + chunks[i]['text']
                            for i, label in source_labels.items())
        for c in arows:
            if c['phase'].startswith(f'decision-{step:02d}-'):
                user = c['forwarded']['messages'][1]['content']
                assert user.endswith(union)
                history = json.loads(user.split('OBSERVATIONS IN ORDER (tool answers are unverified):\n', 1)[1]
                                     .split('\nACCUMULATED RAW SOURCE PASSAGES:', 1)[0])
                assert len(history) == step and history[-1]['query'] == observation['query']
                assert history[-1]['unverified_tool_answer'] == answer_text(observation['answers'])
                assert c['started_unix'] >= max(abyid[r['id']]['started_unix'] + abyid[r['id']]['seconds']
                                               for r in observation['llm_calls'])
                checks += 1
    assert source_labels == result['source_labels']
    budget_checks = 0
    for c in arows:
        assert c['forwarded']['max_tokens'] == (4096 if c['phase'].startswith('search-') else 2800)
        if c['phase'] == 'budget-answer':
            assert c['forwarded']['messages'][1]['content'].endswith(union)
            budget_checks += 1
    first = next(c for c in arows if c['phase'] == 'search-01')
    old_first = next(c for c in calls(old_agent) if c['phase'] == 'search-01')
    agent = {**call_metrics(arows), 'research_seconds': result['seconds'],
             'client_seconds_including_startup': astate['finished_unix'] - astate['started_unix'],
             'searches': len(result['observations']), 'raw_sources': len(source_labels),
             'max_searches': 8, 'top_k_per_search': 15,
             'prior_answers_seeded': False, 'manual_followups': False,
             'fresh_session_per_search': True, 'question_sha256': ap['question_sha256'],
             'outcome': result['stop_reason'], 'action_response_format': ap['action_response_format'],
             'protocol_corrections': len(list((aroot / 'steps').glob('*invalid-action*.json'))),
             'initial_reader_messages_identical_to_native_inkling': first['forwarded']['messages'] == old_first['forwarded']['messages'],
             'answer_words': len(result['final_answer'].split()),
             'audit': {**aaudit, 'latest_answer_and_source_union_checks': checks,
                       'budget_answer_source_union_checks': budget_checks,
                       'graph_counts_unchanged': True, 'policy_and_action_schema_unchanged': True},
             'source_review': load(root / 'agent-review-public.json')}
    unbounded_root = trial / 'inkling-dspark-20261122'
    interruption = load(unbounded_root / 'interruption.json')
    unbounded = {'outcome': 'operator-stopped nonterminating generation',
                 'completed_questions': interruption['completed_questions'],
                 'interrupted_question': interruption['interrupted_question'],
                 'generated_tokens_observed_at_stop': interruption['slots_before_stop'][0]['next_token'][0]['n_decoded'],
                 'whole_attempt_seconds_until_interruption': interruption['stopped_unix'] - load(unbounded_root / 'control.json')['started_unix'],
                 'no_complete_benchmark_score': True}
    output = {'experiment': 'Cognee on Unsloth Inkling Mini with DSpark draft three', 'complete': True,
              'scope': 'Separate from AtlasKV and synthetic decode probes.',
              'backend': 'http://192.168.1.121:8004/v1', 'reader_and_agent_model': 'inkling-2-mini',
              'runtime': 'Unsloth llama.cpp', 'draft_max': 3, 'context_tokens': 65536,
              'parallel_requests': 1, 'prompt_caching': False, 'reasoning_effort': 0.0,
              'graph_counts': counts, 'source_manifest_sha256': sha(source / 'manifest.json'),
              'top_k': 15, 'auto_feedback': False,
              'reasoning_effort_location': 'chat_template_kwargs',
              'thinking_channel_token_blocked': 200008,
              'ingestion_model': manifest['model'], 'ingestion_calls': 0,
              'embedding_model': manifest['embedding_model'], 'embedding_digest': manifest['embedding_digest'],
              'assets': load(Path(__file__).with_name('results-20261121.json'))['assets'],
              'unbounded_attempt': unbounded, 'factual': factual, 'agent': agent,
              'operations': {'protected_files_hash_verified': len(protected),
                             'clef_unchanged': True, 'same_inkling_process_throughout': True,
                             'clef': after['spark2']['clef'], 'memory_guard_tripped': False,
                             'qwen_health_after': after['qwen']['status']},
              'limits': ['Single pass of factual questions and one agent run; no repeated-run confidence intervals.',
                         'Graph and summaries were ingested by Qwen; this tests generation and retrieval, not ingestion.',
                         'Runtime, quantization, prompt caching, device load and dates differ from historical baselines.',
                         'After an unbounded stall, factual readers received a new 4096-token cap; capped generations count as failures.',
                         'Thinking-channel generation is blocked by a model-specific logit ban; this differs from historical native reasoning.',
                         'Source review is Codex assessment, not blinded human adjudication.',
                         'Client worker memory was bounded to 12 GiB; Clef shared Spark 2.']}
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'factual': {k: factual[k] for k in ('recall_seconds', 'recall_p50_seconds', 'recall_p95_seconds', 'quality', 'audit')},
                      'agent': {k: agent[k] for k in ('searches', 'llm_calls', 'client_seconds_including_startup', 'audit')}}, indent=2))


if __name__ == '__main__':
    main()
