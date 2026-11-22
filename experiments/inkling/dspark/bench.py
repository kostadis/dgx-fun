"""Fixed-prompt speed and API checks for Inkling vLLM and Unsloth DSpark."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def schema(name, fields):
    return {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True,
        'schema': {'type': 'object', 'properties': fields, 'required': list(fields),
                   'additionalProperties': False}}}


def fixtures():
    records = [{'name': f'Meria {i:02}', 'home': f'Harbor {i % 7}',
                'item': f'copper key {i:02}', 'price': i+11} for i in range(32)]
    person = {'type': 'object', 'properties': {'name': {'type': 'string'},
        'home': {'type': 'string'}, 'item': {'type': 'string'}, 'price': {'type': 'integer'}},
        'required': ['name', 'home', 'item', 'price'], 'additionalProperties': False}
    return [
        {'id': 'prose', 'prompt': 'Write a detailed field guide to an invented coastal town called Asterwick. '
            'Cover its harbor, markets, council, trades, weather, traditions, transport, buildings, '
            'festivals, and local disputes. Use concrete details and numbered sections. '
            'Aim for at least 1500 words. Begin the guide immediately.'},
        {'id': 'code', 'prompt': 'Write a complete Python module implementing an SQLite-backed task queue. '
            'Include schema creation, atomic job claiming, lease expiry, retries, backoff, idempotency, '
            'input validation, a command line interface, docstrings, and usage examples. '
            'Produce at least 150 lines. Begin with Python code, without explaining your reasoning.'},
        {'id': 'structured', 'prompt': 'Extract every record below in original order. Keep all values exact. '
            'Return an object with a records array, with name, home, item, and price in each entry.\n'
            + '\n'.join(f'{r["name"]} lives in {r["home"]}, carries {r["item"]}, priced at {r["price"]} copper.' for r in records),
         'response_format': schema('records', {'records': {'type': 'array', 'items': person}})},
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='inkling-2-mini')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--output-tokens', type=int, default=512)
    a = p.parse_args();a.output.mkdir(parents=True)
    cases = fixtures()
    fixture_bytes = json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()
    protocol = {'arm': a.arm, 'base': a.base, 'model': a.model, 'rounds': a.rounds,
        'output_budget': a.output_tokens, 'temperature': 0, 'seed': 604, 'reasoning_effort': 'none',
        'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'speed_outputs_are_budget_bounded_not_quality_answers': True,
        'throughput_definition': 'API completion tokens divided by whole request wall time; includes prefill.',
        'stream_timing_note': 'SSE events may contain multiple speculative tokens; event gaps are not per-token ITL.',
        'gguf_prompt_cache': False if a.arm != 'vllm' else 'original vLLM configuration retained'}
    save(a.output/'protocol.json', protocol)
    save(a.output/'fixtures.json', cases)
    state = {'started_unix': time.time(), 'passed': False, 'stage': 'smoke', 'completed': 0}
    save(a.output/'progress.json', state)

    def get(path):
        with urllib.request.urlopen(a.base+path, timeout=10) as r:
            return r.read().decode()

    def metrics():
        try:
            return '\n'.join(line for line in get('/metrics').splitlines()
                if any(key in line.lower() for key in ('spec', 'draft', 'tokens_predicted')))
        except Exception as error:
            return repr(error)

    assert any(m['id'] == a.model for m in json.loads(get('/v1/models'))['data'])
    (a.output/'metrics-before.txt').write_text(metrics()+'\n')

    def request(prompt, maximum, stream=False, **extra):
        payload = {'model': a.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1, 'seed': 604, 'reasoning_effort': 'none',
            'max_tokens': maximum, 'stream': stream, **extra}
        if a.arm != 'vllm':
            payload['cache_prompt'] = False
        if stream:
            payload['stream_options'] = {'include_usage': True}
        req = urllib.request.Request(a.base+'/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        tick = time.monotonic();events = [];content = [];reasoning = [];first = None;usage = {};timings = {}
        with urllib.request.urlopen(req, timeout=300) as r:
            if not stream:
                response = json.load(r)
                return {'request': payload, 'response': response, 'seconds': time.monotonic()-tick}
            for line in r:
                if not line.startswith(b'data:'):
                    continue
                value = line[5:].strip()
                if value == b'[DONE]':
                    break
                event = json.loads(value);offset = time.monotonic()-tick
                if 'error' in event:
                    raise RuntimeError(event['error'])
                events.append({'offset_seconds': offset, 'data': event})
                usage = event.get('usage') or usage
                timings = event.get('timings') or timings
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    text = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if text or thought:
                        if first is None:first = offset
                    content.append(text);reasoning.append(thought)
        seconds = time.monotonic()-tick
        assert usage.get('completion_tokens', 0) > 0 and first is not None
        finish = [c['finish_reason'] for e in events for c in e['data'].get('choices', []) if c.get('finish_reason')]
        return {'request': payload, 'seconds': seconds, 'first_text_event_seconds': first,
            'usage': usage, 'timings': timings, 'finish_reasons': finish,
            'completion_tokens_per_wall_second': usage['completion_tokens']/seconds,
            'answer': ''.join(content), 'reasoning': ''.join(reasoning), 'events': events}

    smokes = {}
    smokes['arithmetic'] = request('What is 17 multiplied by 19? Reply only with the integer.', 128)
    smokes['arithmetic']['passed'] = smokes['arithmetic']['response']['choices'][0]['message']['content'].strip().strip('.') == '323'
    extraction_schema = schema('permit', {'town': {'type': 'string'}, 'color': {'type': 'string'},
        'coins': {'type': 'integer'}, 'official': {'type': 'string'}})
    smokes['structured'] = request('Extract the town, permit color, price in copper coins, and official. '
        'Source: In Asterwick, official Ilyen Moss requires an ochre permit costing 18 copper coins.',
        256, response_format=extraction_schema)
    try:
        parsed = json.loads(smokes['structured']['response']['choices'][0]['message']['content'])
        smokes['structured']['passed'] = parsed == {'town': 'Asterwick', 'color': 'ochre', 'coins': 18, 'official': 'Ilyen Moss'}
    except (TypeError, json.JSONDecodeError):smokes['structured']['passed'] = False
    negative_schema = schema('source_facts', {'open_day': {'type': 'string'}, 'went_to_cave': {'type': 'boolean'}})
    smokes['negation'] = request('Use only this source. Asterwick harbor is closed on Tuesday and open on Wednesday. '
        'Ilyen Moss never went to the cave. Return open_day and whether Ilyen went_to_cave.',
        256, response_format=negative_schema)
    try:
        parsed = json.loads(smokes['negation']['response']['choices'][0]['message']['content'])
        smokes['negation']['passed'] = parsed == {'open_day': 'Wednesday', 'went_to_cave': False}
    except (TypeError, json.JSONDecodeError):smokes['negation']['passed'] = False
    tool = {'type': 'function', 'function': {'name': 'lookup_weather', 'description': 'Look up weather for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city'], 'additionalProperties': False}}}
    smokes['tool'] = request('Use lookup_weather to get the weather in Asterwick. Call the tool before answering.',
        256, tools=[tool], tool_choice='required')
    calls = smokes['tool']['response']['choices'][0]['message'].get('tool_calls') or []
    try:
        smokes['tool']['passed'] = len(calls) == 1 and calls[0]['function']['name'] == 'lookup_weather' and json.loads(calls[0]['function']['arguments']) == {'city': 'Asterwick'}
    except (KeyError, TypeError, json.JSONDecodeError):smokes['tool']['passed'] = False
    save(a.output/'smokes.json', smokes)
    print('smoke', a.arm, {k: v['passed'] for k,v in smokes.items()}, flush=True)

    # Separate warmup from measured requests; no extra candidate tuning.
    save(a.output/'warmup.json', request(cases[0]['prompt'], 128, stream=True))
    for repeat in range(a.rounds):
        for case in cases:
            state.update(stage='speed', case=case['id'], repeat=repeat)
            save(a.output/'progress.json', state)
            result = request(case['prompt'], a.output_tokens, stream=True,
                             **({'response_format': case['response_format']} if 'response_format' in case else {}))
            save(a.output/f'{case["id"]}-{repeat}.json', result)
            state['completed'] += 1
            print(a.arm, case['id'], repeat, round(result['seconds'], 2),
                  round(result['completion_tokens_per_wall_second'], 2), flush=True)
    (a.output/'metrics-after.txt').write_text(metrics()+'\n')
    state.update(stage='complete', passed=True, smoke_passed=all(r['passed'] for r in smokes.values()),
                 finished_unix=time.time())
    save(a.output/'progress.json', state)


if __name__ == '__main__':
    main()
