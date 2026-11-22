"""Spark-side controller: prepare on CPU, swap only Inkling, then restore it."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--attempt',default='20261121')
    p.add_argument('--skip-encode',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--diagnostics-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();out=root/('run-'+a.attempt)
    state={'stage':'preflight','started_unix':time.time(),'passed':False}
    def save():
        tmp=root/('control-'+a.attempt+'.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(root/('control-'+a.attempt+'.json'))
    def inspect(name):
        v=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        return {'name':name,'id':v['Id'],'image':v['Image'],'running':v['State']['Running'],
                'started':v['State']['StartedAt']}
    def health(port):
        with urllib.request.urlopen(f'http://192.168.1.121:{port}/health',timeout=5) as f:
            assert f.status==200
            return f.read().decode()
    def wait_container(name):
        while True:
            v=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
            state['updated_unix']=time.time();save()
            if not v['State']['Running']:
                return v['State']['ExitCode']
            time.sleep(10)
    image='sha256:f4612b72c529a2e899c008b4ecc970445352cc004f38bbe2afe92507ca43167d'
    common=['--mount',f'type=bind,source={root},target=/work',
        '--env','HF_HOME=/work/cache','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1',
        '--env','TOKENIZERS_PARALLELISM=false','--network','none']
    before={name:inspect(name) for name in ['inkling2mini','clef-card-train']}
    assert all(s['running'] for s in before.values())
    assert json.loads(health(8002))['ready'];health(8004)
    (root/('services-before-'+a.attempt+'.json')).write_text(json.dumps(before,indent=2)+'\n')
    stopped=False
    try:
        if not a.skip_encode:
            state['stage']='encoding-and-clustering';save()
            name='atlaskv-encode-'+a.attempt
            command=['docker','run','-d','--name',name,'--memory','6g','--memory-swap','6g','--cpus','6',
                *common,'--env','OMP_NUM_THREADS=4','--env','OPENBLAS_NUM_THREADS=4',
                '--env','NUMBA_NUM_THREADS=4','--entrypoint','/work/venv/bin/python',image,
                '/work/encode.py','--data','/work/data','--assets','/work/downloads.json']
            subprocess.run(command,check=True)
            state['encoding_returncode']=wait_container(name);save()
            if state['encoding_returncode']!=0:raise RuntimeError('CPU embedding/clustering failed')
        assert (root/'data/embedding-manifest.json').exists()
        state['stage']='swapping-inkling';save()
        subprocess.run(['docker','stop','--timeout','45','inkling2mini'],check=True)
        stopped=True
        mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemAvailable']<44*1024**2:raise MemoryError('Need at least 44 GiB available before this worker')
        state['stage']='benchmarking';save()
        name='atlaskv-run-'+a.attempt
        command=['docker','run','-d','--name',name,'--gpus','all','--memory','40g','--memory-swap','40g',
            '--cpus','8','--shm-size','2g',*common,'--env','OMP_NUM_THREADS=8',
            '--entrypoint','/work/venv/bin/python',image,'-u','/work/run.py',
            '--data','/work/data','--assets','/work/downloads.json','--output','/work/'+out.name]
        if a.smoke_only:command.append('--smoke-only')
        if a.diagnostics_only:command.append('--diagnostics-only')
        subprocess.run(command,check=True)
        state['benchmark_returncode']=wait_container(name);save()
        if state['benchmark_returncode']!=0:raise RuntimeError('AtlasKV worker failed; inspect preserved logs')
        state['passed']=True
    except Exception as error:
        state.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        state['stage']='restoring-inkling' if stopped else 'verifying-services';save()
        if stopped:
            try:
                current=inspect('inkling2mini')
                assert current['id']==before['inkling2mini']['id'] and current['image']==before['inkling2mini']['image']
                subprocess.run(['docker','start','inkling2mini'],check=True)
                deadline=time.monotonic()+900
                while True:
                    try:health(8004);break
                    except Exception:
                        if time.monotonic()>deadline:raise TimeoutError('Inkling restoration readiness timed out')
                        state['updated_unix']=time.time();save();time.sleep(3)
            except Exception as error:
                state.update(restoration_error=repr(error),passed=False)
        try:
            after={name:inspect(name) for name in before}
            assert after['clef-card-train']==before['clef-card-train']
            assert all(after[n]['id']==before[n]['id'] and after[n]['image']==before[n]['image'] and after[n]['running'] for n in before)
            assert json.loads(health(8002))['ready'];health(8004)
            state['services_after']=after;state['restoration_passed']=True
        except Exception as error:state.update(final_health_error=repr(error),passed=False)
        state.update(stage='complete' if state['passed'] else 'failed-restored',finished_unix=time.time());save()
    raise SystemExit(0 if state['passed'] else 1)


if __name__=='__main__':main()
