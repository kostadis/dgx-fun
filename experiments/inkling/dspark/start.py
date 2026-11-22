"""Start one pinned GGUF candidate beside Clef, retaining vLLM for rollback."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def inspect(name):
    x = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
    return {'id': x['Id'], 'image': x['Image'], 'running': x['State']['Running'],
            'started': x['State']['StartedAt']}


def available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/kostadis/inkling-dspark-trial'))
    p.add_argument('--profile', choices=['off', 'dspark-3', 'dspark-7'], required=True)
    p.add_argument('--unit', required=True)
    p.add_argument('--stop-vllm', action='store_true')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=30008)
    a = p.parse_args();root = a.root.resolve()
    assert socket.gethostname() == 'gx10-3e5c'
    receipt = json.loads((root/'download-receipt.json').read_text())
    assert receipt['passed']
    for name, metadata in receipt['files'].items():
        assert (root/'models'/name).stat().st_size == metadata['bytes']
    before = {'clef': inspect('clef-card-train'), 'vllm': inspect('inkling2mini')}
    assert before['clef']['running']
    with urllib.request.urlopen('http://192.168.1.121:8002/health', timeout=5) as r:
        assert json.load(r)['ready']
    for unit in ['inkling-gguf-off', 'inkling-gguf-dspark-3', 'inkling-gguf-dspark-7', 'inkling-gguf-active']:
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0:
            raise RuntimeError(f'Stop the preceding candidate first: {unit}')
    command = [str(root/'runtime/llama-server'), '--model', str(root/'models/Inkling-2-Mini-UD-Q4_K_XL.gguf'),
        '--alias', 'inkling-2-mini', '--host', a.host, '--port', str(a.port),
        '--ctx-size', '65536', '--parallel', '1', '--n-gpu-layers', 'all', '--flash-attn', 'on',
        '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--batch-size', '1024', '--ubatch-size', '256',
        '--threads', '8', '--threads-batch', '8', '--no-cache-prompt',
        '--reasoning-effort', 'none', '--metrics', '--perf']
    if a.profile == 'off':
        command += ['--spec-type', 'none']
    else:
        command += ['--model-draft', str(root/'models/dspark-Inkling-2-Mini-Q8_0.gguf'),
                    '--spec-type', 'draft-dspark', '--spec-draft-n-max', a.profile.split('-')[-1],
                    '--n-gpu-layers-draft', 'all']
    record = {'profile': a.profile, 'unit': a.unit, 'command': command, 'before': before,
              'started_unix': time.time(), 'passed': False,
              'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = root/(a.unit+'-start.json')
    def save():destination.write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        if before['vllm']['running']:
            if not a.stop_vllm:
                raise RuntimeError('Pass --stop-vllm to switch the retained Inkling container')
            subprocess.run(['docker', 'stop', '--timeout', '45', 'inkling2mini'], check=True)
        if available() < 40*1024**3:
            raise MemoryError('Need at least 40 GiB available before loading this candidate')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
            check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            check.bind((a.host, a.port))
        subprocess.run(['systemd-run', '--user', '--collect', '--unit='+a.unit,
            '--property=Restart=no', '--property=MemoryMax=40G', '--property=MemorySwapMax=0',
            '--property=CPUQuota=800%', '--working-directory='+str(root), *command], check=True)
        deadline = time.monotonic()+600
        while True:
            record['host_available_bytes'] = available();save()
            if record['host_available_bytes'] < 12*1024**3:
                raise MemoryError('Host memory floor reached')
            if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', a.unit]).returncode != 0:
                raise RuntimeError('Candidate exited during startup')
            try:
                with urllib.request.urlopen(f'http://{a.host}:{a.port}/health', timeout=3) as r:
                    if r.status == 200:
                        record['health'] = json.load(r)
                        with urllib.request.urlopen(f'http://{a.host}:{a.port}/v1/models', timeout=5) as models:
                            assert any(m['id']=='inkling-2-mini' for m in json.load(models)['data'])
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Candidate readiness timed out')
            time.sleep(3)
        assert inspect('clef-card-train') == before['clef']
        record.update(passed=True, finished_unix=time.time(), startup_seconds=time.time()-record['started_unix'])
        save();print(json.dumps(record), flush=True)
    except Exception as error:
        record.update(error=repr(error), finished_unix=time.time());save()
        subprocess.run(['systemctl', '--user', 'stop', a.unit], check=False)
        if not inspect('inkling2mini')['running']:
            subprocess.run(['docker', 'start', 'inkling2mini'], check=True)
        raise


if __name__ == '__main__':
    main()
