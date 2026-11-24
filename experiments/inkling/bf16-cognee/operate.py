"""Temporarily serve the retained large BF16 container and restore selected FP8."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import capacity as C

BF16 = 'inkling-context-bf16-13g-20261122'


def clef(before):
    assert C.inspect('clef-card-train') == before['clef']
    assert C.request('/health', base='http://192.168.1.121:8002')['ready']


def ready(root, identity, before, context):
    start = time.monotonic()
    while True:
        assert C.inspect(identity)['running'] and C.available() >= 12 * C.GiB
        clef(before)
        C.save(root / 'operation-progress.json', {'container': identity, 'context_target': context,
            'available_bytes': C.available(), 'updated_unix': time.time()})
        try:
            C.request('/health')
            model = next(m for m in C.request('/v1/models')['data'] if m['id'] == 'inkling-2-mini')
            assert model['max_model_len'] == context
            return {'model': model, 'seconds': time.monotonic() - start}
        except OSError:
            if time.monotonic() - start > 1200:
                raise TimeoutError('Native readiness timed out')
            time.sleep(3)


def restore(root):
    before = C.load(root / 'before.json')
    C.stop_guard()
    if C.inspect(before['bf16']['id'])['running']:
        C.run('docker', 'stop', '--timeout', '30', before['bf16']['id'])
    if not C.inspect(before['fp8']['id'])['running']:
        C.run('docker', 'start', before['fp8']['id'])
    original_root = Path('/home/kostadis/inkling-context-20261122')
    C.start_guard(original_root, before['fp8']['id'], 'fp8-13g')
    info = ready(root, before['fp8']['id'], before, 411520)
    current = C.inspect(before['fp8']['id'])
    assert all(current[k] == before['fp8'][k] for k in ['id', 'image', 'name', 'command'])
    clef(before)
    smoke = C.complete([{'role': 'user', 'content': 'Return only the integer result of 17 multiplied by 19.'}],
                       'bf16-benchmark-fp8-restored')
    assert smoke['content'].strip() == '323'
    C.save(root / 'restoration.json', {'passed': True, 'fp8': current, 'bf16': C.inspect(BF16),
        'clef': C.inspect('clef-card-train'), 'guard_active': True, 'ready': info,
        'smoke': smoke, 'available_bytes': C.available(), 'finished_unix': time.time()})


def start(root):
    assert not (root / 'before.json').exists(), 'Preserve an earlier switch receipt'
    before = {'fp8': C.inspect('inkling2mini'), 'bf16': C.inspect(BF16),
              'clef': C.inspect('clef-card-train')}
    assert before['fp8']['running'] and not before['bf16']['running']
    assert before['fp8']['image'] == before['bf16']['image'] == C.IMAGE
    assert C.request('/load')['server_load'] == 0
    fp8 = before['fp8']['command']; bf16 = before['bf16']['command']
    assert fp8[fp8.index('--kv-cache-dtype') + 1] == 'fp8'
    assert bf16[bf16.index('--kv-cache-dtype') + 1] == 'auto'
    normalize = lambda args: [value if index != args.index('--kv-cache-dtype') + 1 else 'CACHE_DTYPE'
                              for index, value in enumerate(args)]
    assert normalize(fp8) == normalize(bf16)
    clef(before); C.save(root / 'before.json', before)
    C.save(root / 'fp8-memory-before-switch.json',
           C.load(Path('/home/kostadis/inkling-context-20261122/fp8-13g-memory.json')))
    state = {'passed': False, 'started_unix': time.time()}
    C.save(root / 'startup.json', state)
    try:
        C.stop_guard(); C.run('docker', 'stop', '--timeout', '45', before['fp8']['id'])
        assert C.available() >= 46 * C.GiB
        C.run('docker', 'start', before['bf16']['id'])
        C.start_guard(root, before['bf16']['id'], 'bf16-cognee')
        info = ready(root, before['bf16']['id'], before, 182144)
        native = C.inspect(BF16)
        assert all(native[k] == before['bf16'][k] for k in ['id', 'image', 'name', 'command'])
        state.update(passed=True, native=native, clef=C.inspect('clef-card-train'), ready=info,
                     startup_seconds=time.time() - state['started_unix'], finished_unix=time.time())
        C.save(root / 'startup.json', state)
    except Exception as error:
        state.update(error=repr(error), finished_unix=time.time()); C.save(root / 'startup.json', state)
        restore(root)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('action', choices=['start', 'validate', 'restore'])
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'start':
        start(root)
    elif a.action == 'restore':
        restore(root)
    else:
        before = C.load(root / 'before.json'); current = C.inspect(BF16)
        assert current == C.load(root / 'startup.json')['native'] and current['running']
        clef(before)
        probe = C.long_probe(root, 'bf16-182k', 182144)
        clef(before)
        C.save(root / 'capacity-validation.json', {'passed': True, 'probe': probe,
            'native': C.inspect(BF16), 'clef': C.inspect('clef-card-train'),
            'memory': C.load(root / 'bf16-cognee-memory.json'), 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
