"""Validate context-capacity receipts and export synthetic, public metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); bundle = json.loads(a.input.read_text())
    result = bundle['result']; selected = result['selected']; probe = selected['long_probe']
    assert result['complete'] and result['clef_unchanged']
    assert result['clef'] == result['before']['clef'] == bundle['final_health']['clef']
    assert result['native']['id'] == selected['id'] == bundle['final_health']['native']['id']
    assert result['native'] == bundle['final_health']['native']
    assert result['native']['image'] == result['before']['native']['image']
    assert result['native']['name'] == '/inkling2mini' and result['native']['running']
    assert not bundle['final_health']['rollback']['running']
    assert bundle['final_health']['rollback']['id'] == result['before']['native']['id']
    assert bundle['final_health']['rollback']['name'] == '/inkling2mini-64k-20261122'
    assert bundle['final_health']['guard_active']
    smoke = bundle['final_health']['post_promotion_smoke']
    assert smoke['content'].strip() == '323' and smoke['finish_reason'] == 'stop'
    assert smoke['reasoning_characters'] == 0
    assert smoke['usage']['completion_tokens_details']['reasoning_tokens'] == 0
    model = selected['validation_ready']['model']; limit = model['max_model_len']
    assert bundle['final_health']['model']['max_model_len'] == limit > 65536
    assert probe['advertised_context'] == limit
    assert limit - 1100 <= probe['tokenized_prompt_tokens'] <= limit - 1024
    assert probe['usage']['prompt_tokens'] == probe['tokenized_prompt_tokens']
    assert probe['passed'] and probe['finish_reason'] == 'stop' and probe['reasoning_characters'] == 0
    assert probe['usage']['completion_tokens_details']['reasoning_tokens'] == 0
    assert json.loads(probe['content']) == probe['expected']
    assert probe['usage']['total_tokens'] <= limit
    assert result['kv_budget_bytes'] == 13 * 1024**3
    assert selected['command'][selected['command'].index('--max-model-len') + 1] == 'auto'
    assert selected['command'][selected['command'].index('--kv-cache-memory-bytes') + 1] == str(result['kv_budget_bytes'])
    assert selected['command'][selected['command'].index('--kv-cache-dtype') + 1] == selected['dtype']
    for name, digest in bundle['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    memories = bundle['memory']
    assert memories
    for memory in memories.values():
        assert not memory['floor_tripped'] and memory['min_available_bytes'] >= 12 * 1024**3
    assert bundle['final_health']['available_bytes'] >= 12 * 1024**3
    output = {'experiment': 'Native Inkling context fitting alongside Clef', 'complete': True,
        'verified_utc': bundle['final_health']['checked_utc'], 'context_tokens': limit,
        'previous_context_tokens': 65536, 'kv_cache_dtype': selected['dtype'],
        'kv_budget_bytes': result['kv_budget_bytes'], 'host_floor_bytes': 12 * 1024**3,
        'native': result['native'], 'rollback': bundle['final_health']['rollback'],
        'clef': result['clef'], 'clef_unchanged': True,
        'source_sha256': bundle['source_sha256'], 'guard_unit': result['guard_unit'],
        'candidate_capacities': [{'profile': c['profile'], 'passed': c['passed'],
            **({'context_tokens': c['ready']['model']['max_model_len'], 'dtype': c['dtype'],
                'startup_seconds': c['ready']['startup_seconds'],
                'smoke': [{k: check[k] for k in ['check', 'passed', 'seconds', 'usage', 'reasoning_characters']}
                          for check in c['smoke']]} if c['passed'] else {'error': c['error']})}
            for c in result['candidates']],
        'long_probe': {k: probe[k] for k in ['advertised_context', 'tokenized_prompt_tokens',
            'output_cap', 'prompt_sha256', 'sentinel_placement', 'passed', 'seconds', 'usage',
            'reasoning_characters', 'finish_reason']},
        'memory': memories, 'final_health': {k: bundle['final_health'][k] for k in
            ['checked_utc', 'available_bytes', 'guard_active', 'qwen_model_ids', 'clef_model_ids']},
        'post_promotion_smoke': {k: smoke[k] for k in ['seconds', 'usage', 'reasoning_characters', 'finish_reason']},
        'limits': ['Automatic fit within a conservative 13 GiB KV budget; not an absolute hardware maximum.',
                   'One synthetic near-limit extraction with three exact codes, not a long-context accuracy benchmark.',
                   'KV precision changes can affect answers; original Cognee speed/quality results are not rerun.',
                   'Shared-host headroom can change with Clef and other activity.']}
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['complete', 'context_tokens', 'kv_cache_dtype', 'long_probe']}, indent=2))


if __name__ == '__main__':
    main()
