"""Wait for verified assets, then run one controlled GGUF profile."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/kostadis/inkling-dspark-trial'))
    p.add_argument('--profile',choices=['off','dspark-3','dspark-7'],required=True)
    a=p.parse_args();root=a.root;unit='inkling-gguf-'+a.profile
    state={'profile':a.profile,'stage':'waiting-for-assets','started_unix':time.time(),'passed':False}
    helpers=['inkling-dspark-download','inkling-dspark-download-http','inkling-dspark-download-normal',
             'inkling-dspark-download-fast-bounded','inkling-dspark-download-roomy']
    def running_helpers():
        active=[]
        for name in helpers:
            result=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
            if result.returncode==0 and json.loads(result.stdout)[0]['State']['Running']:
                active.append(name)
        return active
    def save():
        tmp=root/('stage-'+a.profile+'.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(root/('stage-'+a.profile+'.json'))
    try:
        save();deadline=time.monotonic()+1800
        while not (root/'download-receipt.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Asset verification did not finish')
            state['updated_unix']=time.time();save();time.sleep(5)
        assert json.loads((root/'download-receipt.json').read_text())['passed']
        # Xet may retain a substantial reconstruction buffer until its process
        # exits. Release that working memory before loading the model.
        while running_helpers():
            if time.monotonic()>deadline:raise TimeoutError('Download helper did not exit')
            state['updated_unix']=time.time();save();time.sleep(3)
        state['stage']='starting';save()
        subprocess.run(['/usr/bin/python3',str(root/'start.py'),'--profile',a.profile,'--unit',unit,'--stop-vllm'],check=True)
        subprocess.run(['systemd-run','--user','--collect','--unit='+unit+'-memory',
            '--property=MemoryMax=128M','--property=Restart=no','/usr/bin/python3',str(root/'monitor.py'),
            '--unit',unit,'--output',str(root/(unit+'-memory.json'))],check=True)
        state['stage']='benchmarking';save()
        subprocess.run(['/usr/bin/python3','-u',str(root/'bench.py'),'--base','http://127.0.0.1:30008',
            '--arm',a.profile,'--output',str(root/('bench-'+a.profile+'-20261121'))],check=True)
        state.update(stage='complete',passed=True,finished_unix=time.time());save()
    except Exception as error:
        state.update(stage='failed',error=repr(error),finished_unix=time.time());save()
        subprocess.run(['systemctl','--user','stop',unit],check=False)
        for name in running_helpers():
            subprocess.run(['docker','stop','--timeout','10',name],check=False)
        subprocess.run(['docker','start','inkling2mini'],check=True)
        raise


if __name__=='__main__':main()
