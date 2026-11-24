"""Fit native Inkling context beside Clef, validate it, and retain a rollback."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

GiB = 1024**3
BASE = 'http://192.168.1.121:8004'
GUARD = 'inkling-native-context-memory'
IMAGE = 'sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de'


def run(*args):
    return subprocess.check_output(list(args), text=True).strip()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def inspect(name):
    d = json.loads(run('docker', 'inspect', name))[0]
    return {'id': d['Id'], 'image': d['Image'], 'running': d['State']['Running'],
            'started': d['State']['StartedAt'], 'name': d['Name'], 'command': d['Config']['Cmd']}


def request(path, body=None, timeout=20, base=BASE):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read()
        return json.loads(text) if text else None


def available():
    return next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()
                if l.startswith('MemAvailable:'))


def clef_unchanged(before):
    assert inspect('clef-card-train') == before['clef'], 'Clef identity or start time changed'
    assert request('/health', base='http://192.168.1.121:8002')['ready']


def stop_guard():
    subprocess.run(['systemctl', '--user', 'stop', GUARD], check=False)


def start_guard(root, identity, profile):
    stop_guard()
    run('systemd-run', '--user', '--collect', '--unit=' + GUARD,
        '--property=MemoryMax=128M', '--property=Restart=no', '/usr/bin/python3',
        str(root / 'monitor.py'), '--root', str(root), '--container', identity,
        '--profile', profile)


def wait_ready(root, identity, stage, before):
    started = time.monotonic()
    while True:
        assert inspect(identity)['running'], 'Inkling candidate exited'
        assert available() >= 12 * GiB, 'Host headroom crossed the memory floor'
        clef_unchanged(before)
        save(root / 'progress.json', {'stage': stage, 'updated_unix': time.time(),
                                     'available_bytes': available(), 'container_id': identity})
        try:
            request('/health')
            model = next(m for m in request('/v1/models')['data'] if m['id'] == 'inkling-2-mini')
            return {'model': model, 'startup_seconds': time.monotonic() - started}
        except (OSError, urllib.error.URLError):
            if time.monotonic() - started > 1200:
                raise TimeoutError('Native readiness timed out')
            time.sleep(3)


def complete(messages, salt, json_mode=False, timeout=120):
    body = {'model': 'inkling-2-mini', 'messages': messages, 'reasoning_effort': 'none',
            'chat_template_kwargs': {'enable_thinking': False}, 'bad_words': ['<|content_thinking|>'],
            'temperature': 0, 'seed': 604, 'max_tokens': 128, 'cache_salt': salt}
    if json_mode:
        body['response_format'] = {'type': 'json_object'}
    started = time.monotonic()
    response = request('/v1/chat/completions', body, timeout)
    choice = response['choices'][0]
    message = choice['message']
    assert choice['finish_reason'] == 'stop', 'Generation did not finish naturally'
    assert not (message.get('reasoning_content') or message.get('reasoning'))
    assert (response['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) == 0
    return {'seconds': time.monotonic() - started, 'usage': response['usage'],
            'content': message['content'], 'reasoning_characters': 0, 'finish_reason': 'stop'}


def smoke(root, profile):
    checks = []
    for name, prompt, expected, mode in [
        ('arithmetic', 'Return only the integer result of 17 multiplied by 19.', '323', False),
        ('json', 'Return only a JSON object with key result and integer value 323.', {'result': 323}, True),
        ('negation', 'Source: Arin owns a red key and does not own a blue key. Return JSON with owns_blue set to a boolean based only on the source.', {'owns_blue': False}, True),
    ]:
        result = complete([{'role': 'user', 'content': prompt}], profile + '-' + name, mode)
        answer = json.loads(result['content']) if mode else result['content'].strip()
        result.update(check=name, passed=answer == expected)
        checks.append(result)
        save(root / (profile + '-smoke.json'), checks)
        assert result['passed'], 'Short smoke check failed: ' + name
    return checks


def long_probe(root, profile, limit):
    expected = {'alpha': 'cobalt-7319', 'beta': 'amber-5826', 'gamma': 'silver-9402'}
    filler = 'Archive log entry: Routine inventory inspection completed with no anomalies.\n'

    def messages(repeats):
        text = ('ARCHIVE START\n' + filler * max(1, repeats // 100)
                + '\nRecord alpha: access code cobalt-7319.\n' + filler * (repeats // 2)
                + '\nRecord beta: access code amber-5826.\n' + filler * (repeats // 2)
                + '\nRecord gamma: access code silver-9402.\n' + filler * max(1, repeats // 100)
                + '\nARCHIVE END\nExtract the access codes for records alpha, beta and gamma. '
                  'Return only a JSON object with those three names as keys and their exact codes as values.')
        return [{'role': 'system', 'content': 'Answer using only the supplied archive.'},
                {'role': 'user', 'content': text}]

    def count(msgs):
        return request('/tokenize', {'model': 'inkling-2-mini', 'messages': msgs,
            'add_generation_prompt': True, 'chat_template_kwargs':
                {'enable_thinking': False, 'reasoning_effort': 'none'}}, timeout=120)['count']

    target = limit - 1024
    low, high = 1, max(2, target // 8)
    while count(messages(high)) <= target:
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if count(messages(middle)) <= target:
            low = middle
        else:
            high = middle
    msgs = messages(low)
    tokens = count(msgs)
    assert limit - 1100 <= tokens <= target and tokens > 65536
    metadata = {'advertised_context': limit, 'tokenized_prompt_tokens': tokens,
                'output_cap': 128, 'expected': expected,
                'prompt_sha256': hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest(),
                'sentinel_placement': 'approximately 1%, 50%, and 99% of the archive',
                'started_unix': time.time()}
    save(root / (profile + '-long-probe.json'), metadata)
    save(root / 'progress.json', {'stage': profile + '-long-probe', **metadata})
    result = complete(msgs, profile + '-long-context', True, timeout=1800)
    assert result['usage']['prompt_tokens'] == tokens, 'Token counting and actual chat prompt differed'
    result['passed'] = json.loads(result['content']) == expected
    metadata.update(result, finished_unix=time.time())
    save(root / (profile + '-long-probe.json'), metadata)
    assert result['passed'], 'Near-limit archive extraction failed'
    return metadata


def restore(root):
    before = load(root / 'before.json')
    stop_guard()
    for path in root.glob('*-candidate.json'):
        identity = load(path)['id']
        if inspect(identity)['running']:
            run('docker', 'stop', '--timeout', '30', identity)
    old = inspect(before['native']['id'])
    if old['name'] != '/inkling2mini':
        occupied = subprocess.run(['docker', 'inspect', 'inkling2mini'], capture_output=True, text=True)
        if occupied.returncode == 0:
            candidate = json.loads(occupied.stdout)[0]
            assert candidate['Id'] != old['id']
            run('docker', 'rename', candidate['Id'], 'inkling-context-retained-' + candidate['Id'][:12])
        run('docker', 'rename', old['id'], 'inkling2mini')
    if not old['running']:
        run('docker', 'start', old['id'])
    run('systemd-run', '--user', '--collect', '--unit=inkling-native-thinking-memory',
        '--setenv=INKLING_NATIVE_THINKING_ROOT=/home/kostadis/inkling-native-active-20261122',
        '--property=MemoryMax=128M', '--property=Restart=no', '/usr/bin/python3',
        '/home/kostadis/inkling-native-active-20261122/monitor.py')
    ready = wait_ready(root, old['id'], 'restoring-64k', before)
    clef_unchanged(before)
    save(root / 'restoration.json', {'passed': True, **ready, 'native': inspect(old['id'])})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('action', choices=['run', 'restore'])
    a = p.parse_args(); root = a.root.resolve(); root.mkdir(exist_ok=True)
    if a.action == 'restore':
        restore(root); return
    assert not (root / 'before.json').exists(), 'Preserve previous receipts; use a new root'
    before = {'native': inspect('inkling2mini'), 'clef': inspect('clef-card-train')}
    assert before['native']['running'] and before['native']['image'] == IMAGE
    assert request('/load')['server_load'] == 0, 'Inkling has active client requests'
    clef_unchanged(before); save(root / 'before.json', before)
    outcomes = []
    try:
        run('systemctl', '--user', 'stop', 'inkling-native-thinking-memory')
        run('docker', 'stop', '--timeout', '45', before['native']['id'])
        assert available() >= 46 * GiB
        for dtype in ['auto', 'fp8']:
            profile = 'bf16-13g' if dtype == 'auto' else 'fp8-13g'
            identity = None
            try:
                command = list(before['native']['command'])
                command[command.index('--max-model-len') + 1] = 'auto'
                command.extend(['--kv-cache-memory-bytes', str(13 * GiB), '--kv-cache-dtype', dtype])
                name = 'inkling-context-' + profile + '-20261122'
                identity = run('docker', 'run', '-d', '--name', name, '--gpus', 'all', '--shm-size', '8g',
                    '--memory', '44g', '--memory-swap', '44g', '--cpus', '8', '-p', '192.168.1.121:8004:8000',
                    '-v', '/home/kostadis/inkling-trial/cache:/hf', '-e', 'HF_HOME=/hf',
                    '-v', '/home/kostadis/inkling-trial/runtime:/work', '-e', 'VLLM_CACHE_ROOT=/work/vllm-cache',
                    IMAGE, *command)
                save(root / (profile + '-candidate.json'), {'id': identity, 'command': command, 'dtype': dtype})
                start_guard(root, identity, profile)
                ready = wait_ready(root, identity, profile + '-starting', before)
                checks = smoke(root, profile)
                outcome = {'profile': profile, 'dtype': dtype, 'id': identity, 'command': command,
                           'ready': ready, 'smoke': checks, 'passed': True}
            except Exception as error:
                outcome = {'profile': profile, 'id': identity, 'passed': False, 'error': repr(error)}
            outcomes.append(outcome); save(root / 'candidates.json', outcomes)
            if identity:
                if not (dtype == 'fp8' and outcome['passed']):
                    stop_guard()
                    if inspect(identity)['running']:
                        run('docker', 'stop', '--timeout', '30', identity)
                logs = subprocess.run(['docker', 'logs', identity], capture_output=True, text=True)
                (root / (profile + '.log')).write_text(logs.stdout + logs.stderr)

        choices = sorted((o for o in outcomes if o['passed']),
                         key=lambda o: o['ready']['model']['max_model_len'], reverse=True)
        for selected in choices:
            identity = selected['id']; profile = selected['profile']
            try:
                if not inspect(identity)['running']:
                    assert available() >= 46 * GiB
                    run('docker', 'start', identity)
                    start_guard(root, identity, profile + '-validation')
                ready = wait_ready(root, identity, profile + '-validation-starting', before)
                limit = ready['model']['max_model_len']
                probe = long_probe(root, profile, limit)
                assert available() >= 12 * GiB
                clef_unchanged(before)
                run('docker', 'rename', before['native']['id'], 'inkling2mini-64k-20261122')
                run('docker', 'rename', identity, 'inkling2mini')
                selected.update(long_probe=probe, validation_ready=ready)
                save(root / 'result.json', {'complete': True, 'selected': selected,
                    'native': inspect(identity), 'clef': inspect('clef-card-train'),
                    'clef_unchanged': True, 'available_bytes': available(), 'kv_budget_bytes': 13 * GiB,
                    'guard_unit': GUARD, 'before': before, 'candidates': outcomes, 'finished_unix': time.time()})
                save(root / 'progress.json', {'stage': 'complete', 'complete': True,
                    'context_tokens': limit, 'dtype': selected['dtype'], 'updated_unix': time.time()})
                return
            except Exception as error:
                save(root / (profile + '-validation-error.json'), {'error': repr(error)})
                stop_guard()
                if inspect(identity)['running']:
                    run('docker', 'stop', '--timeout', '30', identity)
        raise RuntimeError('No larger context passed the guarded validation')
    except Exception as error:
        save(root / 'failure.json', {'error': repr(error), 'unix': time.time()})
        restore(root)
        raise


if __name__ == '__main__':
    main()
