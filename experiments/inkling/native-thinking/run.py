"""Paired native Inkling Cognee replay; restore the prior DSpark endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import shlex
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[3]
SSH = ['ssh', '-F', '/home/kostadis/.ssh/config', 'spark2']
REMOTE = '/home/kostadis/inkling-native-thinking-20261122'
BASE = 'http://192.168.1.121:8004'
PYTHON = '/home/kostadis/cognee-local/.venv/bin/python'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def request(path, body=None, method='GET'):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body or {}).encode() if method == 'POST' else None,
                                 headers={'Content-Type': 'application/json'})
    tick = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        text = r.read().decode()
        value = json.loads(text) if text else None
        return {'response': value, 'seconds': time.monotonic() - tick, 'status': r.status}


def remote_json(name):
    return json.loads(subprocess.check_output([*SSH, 'cat ' + shlex.quote(REMOTE + '/' + name)], text=True))


def main():
    global REMOTE
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', default=REMOTE)
    a = p.parse_args(); root = a.output.resolve(); trial = root.parent
    REMOTE = a.remote_root
    assert root.is_relative_to(REPO / 'experiments/adaptive-chunking/raw')
    root.mkdir(exist_ok=True)
    assert not (root / 'control.json').exists(), 'Use a fresh output'
    source = trial / 'adaptive-context-8'
    state = {'started_unix': time.time(), 'stage': 'waiting-for-native', 'complete': False}
    def record(): save(root / 'control.json', state)
    def interrupt(signum, frame): raise InterruptedError('Benchmark interrupted; restoring DSpark')
    signal.signal(signal.SIGTERM, interrupt)
    record()
    try:
        deadline = time.monotonic() + 1400
        while True:
            try:
                startup = remote_json('startup.json')
                save(root / 'native-startup.json', startup)
                if startup.get('passed'):
                    break
                if startup.get('error'):
                    raise RuntimeError(startup['error'])
            except subprocess.CalledProcessError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Native startup did not complete')
            time.sleep(5)
        before = remote_json('before.json'); save(root / 'services-before.json', before)
        assert startup['native']['image'] == before['native']['image']
        assert startup['native']['command'] == before['native']['command']
        protected = {str(p.relative_to(trial)): sha(p) for p in trial.rglob('*.json')
                     if not p.is_relative_to(root)}
        for name in ('data', 'system', 'corpus', 'chunks'):
            for f in (source / name).rglob('*'):
                if f.is_file(): protected[str(f.relative_to(trial))] = sha(f)
        save(root / 'protected-inputs.json', protected)
        options = {(arm, kind): {'cache_salt': root.name + '-' + arm + '-' + kind,
                                **({'bad_words': ['<|content_thinking|>']} if arm == 'off' else {})}
                   for arm in ('on', 'off') for kind in ('questions', 'agent')}
        for (arm, kind), value in options.items(): save(root / (arm + '-' + kind + '-options.json'), value)
        save(root / 'cache-isolation.json', {'method': 'Separate cache_salt per arm and phase',
                                            'salts': {arm + '-' + kind: v['cache_salt'] for (arm, kind), v in options.items()},
                                            'same_native_process': True, 'global_cache_reset': False})
        smokes = load(REPO / 'experiments/inkling/dspark/raw/20261121/bench-vllm-20261121/smokes.json')

        def warmup(arm, directory):
            records = {}
            for name, previous in smokes.items():
                body = {**previous['request'], **options[(arm, 'questions')]}
                item = request('/v1/chat/completions', body, 'POST')
                response = item['response']; message = response['choices'][0]['message']
                if name == 'arithmetic':
                    passed = message['content'].strip().strip('.') == '323'
                elif name == 'structured':
                    passed = json.loads(message['content']) == {'town': 'Asterwick', 'color': 'ochre', 'coins': 18, 'official': 'Ilyen Moss'}
                elif name == 'negation':
                    passed = json.loads(message['content']) == {'open_day': 'Wednesday', 'went_to_cave': False}
                else:
                    calls = message.get('tool_calls') or []
                    passed = len(calls) == 1 and calls[0]['function']['name'] == 'lookup_weather' and json.loads(calls[0]['function']['arguments']) == {'city': 'Asterwick'}
                if arm == 'off':
                    assert not message.get('reasoning_content')
                    assert (response['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) == 0
                records[name] = {'request': body, **item, 'passed': passed}
                save(directory / 'warmup.json', records)
                assert passed and response['choices'][0]['finish_reason'] in ('stop', 'tool_calls')

        # Verify the native hard ban before spending time on the control arm.
        body = {**smokes['arithmetic']['request'], 'cache_salt': root.name + '-preflight', 'bad_words': ['<|content_thinking|>']}
        preflight = request('/v1/chat/completions', body, 'POST')
        save(root / 'no-thinking-preflight.json', {'request': body, **preflight})
        assert preflight['response']['choices'][0]['message']['content'].strip() == '323'
        assert not preflight['response']['choices'][0]['message'].get('reasoning_content')
        assert (preflight['response']['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) == 0
        commands = {}
        for arm in ('on', 'off'):
            directory = root / arm; directory.mkdir()
            state.update(stage=arm + '-warmup'); record()
            warmup(arm, directory)
            common = ['--source', str(source), '--backend', BASE, '--model', 'inkling-2-mini',
                      '--reasoning-effort', 'none']
            commands[arm] = {
                'questions': [PYTHON, str(REPO / 'experiments/adaptive-chunking/cognee/questions.py'), *common,
                              '--extra-body-file', str(root / (arm + '-questions-options.json')),
                              '--output', str(directory / 'questions'), '--cases', str(trial / 'frozen/cases.json'),
                              '--max-output-tokens', '4096', '--continue-on-truncation'],
                'agent': [PYTHON, str(REPO / 'experiments/adaptive-chunking/cognee/sequential.py'), *common,
                          '--extra-body-file', str(root / (arm + '-agent-options.json')),
                          '--output', str(directory / 'agent'), '--question-file', str(trial / 'sonnet-sequential-20261121/question.json'),
                          '--provider', 'openai', '--action-format', 'json_schema', '--top-k', '15', '--max-searches', '8',
                          '--reader-max-output-tokens', '4096']}
            save(root / 'commands.json', commands)
            for kind, command in commands[arm].items():
                state.update(stage=arm + '-' + kind); record()
                with (directory / (kind + '.log')).open('w') as log:
                    result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
                state[arm + '-' + kind + '-returncode'] = result.returncode; record()
                request('/health')
                memory = remote_json('memory.json'); save(directory / (kind + '-memory.json'), memory)
                assert not memory['floor_tripped']
            save(directory / 'memory-end.json', remote_json('memory.json'))
        state.update(stage='verifying-protected-inputs'); record()
        mismatches = [name for name, value in protected.items() if not (trial / name).is_file() or sha(trial / name) != value]
        save(root / 'protected-input-audit.json', {'files': len(protected), 'unchanged': not mismatches, 'mismatches': mismatches})
        assert not mismatches
        state['measurements_complete'] = True
    except Exception as error:
        state['error'] = repr(error)
    finally:
        state['stage'] = 'restoring-dspark'; record()
        with (root / 'restoration.log').open('w') as log:
            result = subprocess.run([*SSH, 'env INKLING_NATIVE_THINKING_ROOT=' + shlex.quote(REMOTE) +
                                     ' /usr/bin/python3 ' + shlex.quote(REMOTE + '/operate.py') + ' restore'],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        state['restoration_returncode'] = result.returncode
        restoration = remote_json('restoration.json'); save(root / 'restoration.json', restoration)
        state['complete'] = bool(state.get('measurements_complete') and result.returncode == 0 and restoration['passed'])
        state.update(stage='complete-awaiting-review' if state['complete'] else 'failed', finished_unix=time.time()); record()
    raise SystemExit(0 if state['complete'] else 1)


if __name__ == '__main__': main()
