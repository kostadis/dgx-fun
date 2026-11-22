"""Validate a completed comparison and export metrics without module text."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


ARMS = ('rag', 'atlas', 'pruned')


def load(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def distribution(values):
    return {'sum': sum(values), 'p50': statistics.median(values),
            'p95': percentile(values, .95), 'min': min(values), 'max': max(values)}


def analyze(run, data, review=None):
    protocol = load(run / 'protocol.json')
    progress = load(run / 'progress.json')
    cases = load(data / 'cases.json')
    ids = [case['id'] for case in cases] + ['narrative']
    questions = {case['id']: case['question'] for case in cases}
    questions['narrative'] = load(data / 'question.json')['question']
    assert len(set(ids)) == len(ids) == 49
    assert progress['passed'] and progress['stage'] == 'complete'
    assert progress['completed'] == {arm: len(ids) for arm in ARMS}
    assert hashlib.sha256((data / 'facts.json').read_bytes()).hexdigest() == protocol['facts_sha256']
    assert hashlib.sha256((data / 'cases.json').read_bytes()).hexdigest() == protocol['cases_sha256']
    count = protocol['facts']
    facts = load(data / 'facts.json')
    assert protocol['attention_configuration']['effective_inner'] == 'eager'
    assert load(run / 'sentence-encoder-audit.json')['identical']
    assert load(run / 'text-grounding-smoke.json')['contains_expected']
    metrics = {}
    for arm in ARMS:
        rows = {load(p)['id']: load(p) for p in sorted((run / arm).glob('*.json'))
                if p.name != 'setup.json'}
        assert set(rows) == set(ids)
        for case_id, row in rows.items():
            assert row['arm'] == arm and row['question'] == questions[case_id]
            assert row['facts_sha256'] == protocol['facts_sha256']
            assert len(row['token_offsets_seconds']) == row['output_tokens'] > 0
            assert 0 <= row['ttft_first_generated_token_seconds'] <= row['seconds']
            assert row['token_offsets_seconds'] == sorted(row['token_offsets_seconds'])
            assert row['token_offsets_seconds'][-1] <= row['seconds']
            assert row['finish_reason'] in {'stop', 'length'}
            for field in ('retrieved_fact_indices', 'pruned_selected_fact_indices',
                          'pruned_prefill_selected_fact_indices'):
                assert all(0 <= index < count for index in row[field])
            if arm == 'rag':
                assert len(row['retrieved_fact_indices']) == 15
                assert not row['all_facts_available'] and not row['pruned_selected_fact_indices']
                for index in row['retrieved_fact_indices']:
                    fact = facts[index]
                    assert f'[{fact["label"]}] {fact["key_string"]} is {fact["description"]}.' in row['prompt']
            elif arm == 'atlas':
                assert row['all_facts_available'] and not row['retrieved_fact_indices']
            else:
                # The pinned 32-layer checkpoint injects at 0,3,...,30. Each
                # forward performs root/intermediate/leaf selection per layer.
                assert row['selection_calls'] == row['output_tokens'] * 11 * 3
                assert 0 < len(row['pruned_prefill_selected_fact_indices']) <= 11 * 16
                assert set(row['pruned_prefill_selected_fact_indices']) <= set(row['pruned_selected_fact_indices'])
                assert not row['all_facts_available'] and not row['retrieved_fact_indices']
        factual = [rows[c['id']] for c in cases]
        samples = [sample for row in rows.values() for sample in row['memory_samples']]
        intervals = [b-a for row in factual for a,b in zip(
            row['token_offsets_seconds'], row['token_offsets_seconds'][1:])]
        narrative = rows['narrative']
        metrics[arm] = {
            'factual_questions': len(factual),
            'factual_seconds': distribution([r['seconds'] for r in factual]),
            'factual_ttft_seconds': distribution([r['ttft_first_generated_token_seconds'] for r in factual]),
            'factual_inter_token_seconds': distribution(intervals),
            'factual_output_tokens': sum(r['output_tokens'] for r in factual),
            'factual_text_prompt_tokens': sum(r['prompt_tokens'] for r in factual),
            'factual_finish_reasons': dict(Counter(r['finish_reason'] for r in factual)),
            'narrative': {key: narrative[key] for key in ('seconds', 'ttft_first_generated_token_seconds',
                                                       'prompt_tokens', 'output_tokens', 'finish_reason')},
            'setup_seconds': load(run / arm / 'setup.json')['seconds'],
            'worker_cuda_peak_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in rows.values()),
            'worker_cuda_peak_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in rows.values()),
            'host_min_available_bytes': min(s['host_available_bytes'] for s in samples),
            'host_max_swap_used_bytes': max(s['host_swap_used_bytes'] for s in samples),
            'worker_max_process_rss_bytes': max(s['process_rss_bytes'] for s in samples),
            'worker_max_cgroup_bytes': max(s['cgroup_current_bytes'] for s in samples if s['cgroup_current_bytes'] is not None),
            'per_question': [{key: rows[case_id][key] for key in (
                'id', 'seconds', 'ttft_first_generated_token_seconds', 'prompt_tokens',
                'output_tokens', 'finish_reason')} for case_id in ids],
        }
    result = {
        'experiment': 'AtlasKV standalone Phandelver fact-memory trial',
        'scope': 'Local controls for the released Llama/MiniLM bundle; separate from prior whole-system and paper-reported results.',
        'complete': True, 'protocol': {key: protocol[key] for key in (
            'facts', 'facts_sha256', 'cases_sha256', 'runner_sha256', 'seed',
            'do_sample', 'top_k_text_retrieval', 'hierarchy_top_k', 'kb_scale_factor',
            'kb_layer_frequency', 'same_checkpoint_for_all_arms', 'runtime', 'attention_configuration')},
        'assets': {key: {k: protocol[key][k] for k in ('repo', 'revision')} for key in (
            'model', 'encoder', 'sentence_encoder', 'tokenizer')},
        'elapsed_worker_seconds': progress['finished_unix'] - progress['started_unix'],
        'arms': metrics,
    }
    if review is not None:
        reviewed = load(review)
        assert set(reviewed) == set(ARMS)
        allowed = {'correct_and_complete', 'partial', 'incorrect', 'abstained', 'source_ambiguous'}
        grounding = {'supported', 'unsupported', 'unassessed'}
        result['quality'] = {}
        for arm in ARMS:
            assert set(reviewed[arm]) == set(questions) - {'narrative'}
            entries = reviewed[arm]
            for case_id, entry in entries.items():
                assert entry['correctness'] in allowed and entry['grounding'] in grounding
                assert (entry['correctness'] == 'source_ambiguous') == (case_id == 'q37')
            result['quality'][arm] = {
                'adjudicable': 47,
                'correctness': dict(Counter(v['correctness'] for v in entries.values())),
                'grounding': dict(Counter(v['grounding'] for v in entries.values())),
                'grounded_complete': sum(v['correctness'] == 'correct_and_complete' and
                                         v['grounding'] == 'supported' for v in entries.values()),
                'labels': {case_id: {key: entry[key] for key in ('correctness', 'grounding')}
                           for case_id, entry in entries.items()},
            }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--review', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(analyze(a.run, a.data, a.review), indent=2) + '\n')


if __name__ == '__main__':
    main()
