"""Replay Cognee on retained 182K BF16 Inkling and restore selected FP8 afterward."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[3]
PYTHON = '/home/kostadis/cognee-local/.venv/bin/python'
BASE = 'http://192.168.1.121:8004'
SSH = ['ssh', '-F', '/home/kostadis/.ssh/config', 'spark2']
REMOTE = '/home/kostadis/inkling-bf16-cognee-20261122'


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def services():
    script = '''import json,subprocess,urllib.request
from pathlib import Path
out={}
for key,name in [('native','inkling-context-bf16-13g-20261122'),('clef','clef-card-train')]:
 d=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
 out[key]={'id':d['Id'],'image':d['Image'],'running':d['State']['Running'],'started':d['State']['StartedAt'],'name':d['Name'],'command':d['Config']['Cmd']}
for key,url in [('models','http://192.168.1.121:8004/v1/models'),('load','http://192.168.1.121:8004/load'),('clef_health','http://192.168.1.121:8002/health')]:
 with urllib.request.urlopen(url,timeout=15) as r:out[key]=json.load(r)
out['guard_active']=subprocess.run(['systemctl','--user','is-active','--quiet','inkling-native-context-memory']).returncode==0
out['guard_stats']=json.loads(Path('/home/kostadis/inkling-bf16-cognee-20261122/bf16-cognee-memory.json').read_text())
out['available_bytes']=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
print(json.dumps(out))
'''
    response = subprocess.run([*SSH, 'python3 -'], input=script, text=True, capture_output=True,
                              check=True, timeout=45)
    return json.loads(response.stdout)


def check_services(current, before):
    assert current['native'] == before['native'] and current['clef'] == before['clef']
    assert current['clef_health']['ready'] and current['guard_active']
    assert not current['guard_stats']['floor_tripped'] and current['available_bytes'] >= 12 * 1024**3
    assert next(m for m in current['models']['data'] if m['id'] == 'inkling-2-mini')['max_model_len'] == 182144


def remote_json(name):
    return json.loads(subprocess.check_output([*SSH, 'cat ' + REMOTE + '/' + name], text=True))


def remote_action(action, output, timeout):
    with output.open('w') as log:
        return subprocess.run([*SSH, '/usr/bin/python3 ' + REMOTE + '/operate.py --root ' + REMOTE + ' ' + action],
                              stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.output.resolve(); trial = root.parent
    assert root.is_relative_to(REPO / 'experiments/adaptive-chunking/raw')
    root.mkdir(exist_ok=True)
    assert not (root / 'control.json').exists(), 'Use a fresh output directory'
    source = trial / 'adaptive-context-8'
    previous = trial / 'inkling-native-thinking-paired-20261122/off'
    state = {'started_unix': time.time(), 'complete': False, 'stage': 'preflight'}
    def interrupted(signum, frame):
        raise InterruptedError('Benchmark interrupted; restoring selected FP8')
    signal.signal(signal.SIGTERM, interrupted)
    save(root / 'control.json', state)
    before = None
    try:
        state['stage'] = 'waiting-for-bf16'; save(root / 'control.json', state)
        deadline = time.monotonic() + 1400
        while True:
            startup = remote_json('startup.json')
            if startup.get('passed'):
                break
            if startup.get('error'):
                raise RuntimeError(startup['error'])
            if time.monotonic() > deadline:
                raise TimeoutError('BF16 startup did not complete')
            time.sleep(3)
        save(root / 'native-startup.json', startup)
        save(root / 'switch-before.json', remote_json('before.json'))
        save(root / 'operation-source-sha256.json', remote_json('source-sha256.json'))
        before = services(); save(root / 'services-before.json', before)
        assert before['native'] == startup['native']
        assert before['load']['server_load'] == 0
        check_services(before, before)
        command = before['native']['command']
        assert command[command.index('--kv-cache-dtype') + 1] == 'auto'
        assert command[command.index('--kv-cache-memory-bytes') + 1] == '13958643712'
        protected = {str(f.relative_to(trial)): sha(f) for f in trial.rglob('*.json')
                     if not f.is_relative_to(root)}
        for name in ('data', 'system', 'corpus', 'chunks'):
            for f in (source / name).rglob('*'):
                if f.is_file():
                    protected[str(f.relative_to(trial))] = sha(f)
        save(root / 'protected-inputs.json', protected)
        snapshots = [Path(__file__), REPO / 'experiments/adaptive-chunking/cognee/questions.py',
                     REPO / 'experiments/adaptive-chunking/cognee/sequential.py']
        hashes = {}
        for f in snapshots:
            target = root / (f.parent.name + '-' + f.name + '.snapshot')
            target.write_bytes(f.read_bytes()); hashes[str(f.relative_to(REPO))] = sha(f)
        save(root / 'source-sha256.json', hashes)
        options = {kind: {'bad_words': ['<|content_thinking|>'],
                         'cache_salt': root.name + '-' + kind} for kind in ('questions', 'agent')}
        for kind, value in options.items():
            save(root / (kind + '-options.json'), value)
        warmups = {}
        state['stage'] = 'warmup'; save(root / 'control.json', state)
        for name, old in load(previous / 'warmup.json').items():
            body = {**old['request'], **options['questions']}
            req = urllib.request.Request(BASE + '/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            tick = time.monotonic()
            with urllib.request.urlopen(req, timeout=120) as response:
                value = json.load(response)
            message = value['choices'][0]['message']
            assert not (message.get('reasoning_content') or message.get('reasoning'))
            assert value['usage']['completion_tokens_details']['reasoning_tokens'] == 0
            if name == 'arithmetic':
                passed = message['content'].strip().strip('.') == '323'
            elif name == 'structured':
                passed = json.loads(message['content']) == {'town': 'Asterwick', 'color': 'ochre', 'coins': 18, 'official': 'Ilyen Moss'}
            elif name == 'negation':
                passed = json.loads(message['content']) == {'open_day': 'Wednesday', 'went_to_cave': False}
            else:
                calls = message.get('tool_calls') or []
                passed = len(calls) == 1 and calls[0]['function']['name'] == 'lookup_weather' and json.loads(calls[0]['function']['arguments']) == {'city': 'Asterwick'}
            warmups[name] = {'request': body, 'response': value, 'seconds': time.monotonic() - tick, 'passed': passed}
            save(root / 'warmup.json', warmups)
            assert passed and value['choices'][0]['finish_reason'] in ('stop', 'tool_calls')
        common = ['--source', str(source), '--backend', BASE, '--model', 'inkling-2-mini', '--reasoning-effort', 'none']
        commands = {
            'questions': [PYTHON, str(REPO / 'experiments/adaptive-chunking/cognee/questions.py'), *common,
                '--extra-body-file', str(root / 'questions-options.json'), '--output', str(root / 'questions'),
                '--cases', str(trial / 'frozen/cases.json'), '--max-output-tokens', '4096', '--continue-on-truncation'],
            'agent': [PYTHON, str(REPO / 'experiments/adaptive-chunking/cognee/sequential.py'), *common,
                '--extra-body-file', str(root / 'agent-options.json'), '--output', str(root / 'agent'),
                '--question-file', str(trial / 'sonnet-sequential-20261121/question.json'),
                '--provider', 'openai', '--action-format', 'json_schema', '--top-k', '15', '--max-searches', '8',
                '--reader-max-output-tokens', '4096']}
        save(root / 'commands.json', commands)
        for kind, command in commands.items():
            state['stage'] = kind; save(root / 'control.json', state)
            with (root / (kind + '.log')).open('w') as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            state[kind + '_returncode'] = result.returncode
            current = services(); save(root / ('services-after-' + kind + '.json'), current)
            check_services(current, before)
            save(root / 'control.json', state)
            assert result.returncode == 0, kind + ' runner did not complete successfully; retain failure'
        mismatches = [name for name, digest in protected.items() if not (trial / name).is_file() or sha(trial / name) != digest]
        save(root / 'protected-input-audit.json', {'files': len(protected), 'unchanged': not mismatches, 'mismatches': mismatches})
        assert not mismatches
        state['stage'] = 'near-limit-bf16-validation'; save(root / 'control.json', state)
        result = remote_action('validate', root / 'capacity-validation.log', 2000)
        state['capacity_returncode'] = result
        assert result == 0, 'BF16 near-limit validation failed'
        validation = remote_json('capacity-validation.json'); save(root / 'capacity-validation.json', validation)
        assert validation['passed']
        state['measurements_complete'] = True
    except Exception as error:
        state['error'] = repr(error)
    finally:
        if before:
            try:
                after = services(); check_services(after, before)
                save(root / 'services-after.json', after); state['services_unchanged'] = True
            except Exception as error:
                state['service_verification_error'] = repr(error); state['complete'] = False
        state['stage'] = 'restoring-selected-fp8'; save(root / 'control.json', state)
        try:
            code = remote_action('restore', root / 'restoration.log', 1400)
            restoration = remote_json('restoration.json'); save(root / 'restoration.json', restoration)
            state['restoration_returncode'] = code
            state['complete'] = bool(state.get('measurements_complete') and state.get('services_unchanged')
                                     and code == 0 and restoration['passed'])
        except Exception as error:
            state['restoration_error'] = repr(error)
        state.update(stage='complete-awaiting-review' if state['complete'] else 'failed', finished_unix=time.time())
        save(root / 'control.json', state)
    raise SystemExit(0 if state['complete'] else 1)


if __name__ == '__main__':
    main()
