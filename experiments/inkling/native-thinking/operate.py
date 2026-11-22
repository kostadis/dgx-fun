"""Switch only Inkling on Spark 2; retain Clef and restore the DSpark service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(os.environ.get('INKLING_NATIVE_THINKING_ROOT', '/home/kostadis/inkling-native-thinking-20261122'))
DSPARK = Path('/home/kostadis/inkling-dspark-trial')
IMAGE = 'sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de'


def save(name, value):
    p = ROOT / name; temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(p)


def inspect(name):
    x = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
    return {'id': x['Id'], 'image': x['Image'], 'running': x['State']['Running'],
            'started': x['State']['StartedAt'], 'command': x['Config']['Cmd']}


def get(path, port=8004):
    with urllib.request.urlopen(f'http://192.168.1.121:{port}{path}', timeout=10) as r:
        body = r.read().decode()
        return json.loads(body) if body else None


def available():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def unchanged_clef(before):
    assert inspect('clef-card-train') == before['clef']
    assert get('/health', 8002)['ready']


def restore():
    before = json.loads((ROOT / 'before.json').read_text())
    state = {'started_unix': time.time(), 'passed': False}
    save('restoration.json', state)
    subprocess.run(['systemctl', '--user', 'stop', 'inkling-native-thinking-memory'], check=False)
    if inspect('inkling2mini')['running']:
        subprocess.run(['docker', 'stop', '--timeout', '45', 'inkling2mini'], check=True)
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'inkling-gguf-active']).returncode:
        subprocess.run(['/usr/bin/python3', str(DSPARK / 'start.py'), '--profile', 'dspark-3',
                        '--unit', 'inkling-gguf-active', '--host', '192.168.1.121', '--port', '8004'], check=True)
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'inkling-gguf-active-memory']).returncode:
        subprocess.run(['systemd-run', '--user', '--collect', '--unit=inkling-gguf-active-memory',
                        '--property=MemoryMax=128M', '--property=Restart=no', '/usr/bin/python3',
                        str(DSPARK / 'monitor.py'), '--unit', 'inkling-gguf-active', '--output',
                        str(DSPARK / 'inkling-gguf-active-memory.json')], check=True)
    unchanged_clef(before)
    current = json.loads((DSPARK / 'inkling-gguf-active-start.json').read_text())
    assert current['command'] == before['dspark_start']['command']
    assert get('/health')['status'] == 'ok'
    assert any(m['id'] == 'inkling-2-mini' for m in get('/v1/models')['data'])
    state.update(passed=True, finished_unix=time.time(), clef=inspect('clef-card-train'),
                 native=inspect('inkling2mini'), dspark_start=current)
    save('restoration.json', state)


def start():
    assert not (ROOT / 'before.json').exists(), 'Use the saved receipt; do not overwrite a prior switch'
    clef, native = inspect('clef-card-train'), inspect('inkling2mini')
    assert clef['running'] and not native['running'] and native['image'] == IMAGE
    assert get('/health', 8002)['ready']
    assert not any(slot['is_processing'] for slot in get('/slots')), 'Inkling has an active client request'
    before = {'clef': clef, 'native': native,
              'dspark_start': json.loads((DSPARK / 'inkling-gguf-active-start.json').read_text()),
              'dspark_memory': json.loads((DSPARK / 'inkling-gguf-active-memory.json').read_text())}
    save('before.json', before)
    state = {'started_unix': time.time(), 'stage': 'stopping-dspark', 'passed': False}
    save('startup.json', state)
    try:
        subprocess.run(['systemctl', '--user', 'stop', 'inkling-gguf-active', 'inkling-gguf-active-memory'], check=True)
        assert available() >= 46 * 1024**3, 'Insufficient headroom for the retained native configuration'
        state['stage'] = 'native-starting'; save('startup.json', state)
        subprocess.run(['docker', 'start', 'inkling2mini'], check=True)
        subprocess.run(['systemd-run', '--user', '--collect', '--unit=inkling-native-thinking-memory',
                        '--setenv=INKLING_NATIVE_THINKING_ROOT=' + str(ROOT),
                        '--property=MemoryMax=128M', '--property=Restart=no', '/usr/bin/python3',
                        str(ROOT / 'monitor.py')], check=True)
        deadline = time.monotonic() + 1200
        while True:
            state['available_bytes'] = available(); state['updated_unix'] = time.time(); save('startup.json', state)
            assert state['available_bytes'] >= 12 * 1024**3
            assert inspect('inkling2mini')['running'], 'Native container exited'
            try:
                get('/health')
                assert any(m['id'] == 'inkling-2-mini' for m in get('/v1/models')['data'])
                break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Native readiness timed out')
            time.sleep(3)
        unchanged_clef(before)
        after = inspect('inkling2mini')
        assert after['id'] == native['id'] and after['image'] == native['image'] and after['command'] == native['command']
        state.update(stage='ready', passed=True, native=after, finished_unix=time.time(),
                     startup_seconds=time.time() - state['started_unix'])
        save('startup.json', state)
    except Exception as error:
        state.update(error=repr(error), finished_unix=time.time()); save('startup.json', state)
        restore()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['start', 'restore'])
    a = p.parse_args(); ROOT.mkdir(exist_ok=True)
    start() if a.action == 'start' else restore()
