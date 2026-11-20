import json,os,signal,subprocess,time
from pathlib import Path
r=Path.home()/'qwen38-dflash-trial'
def bench(label,only=False):
 cmd=['python3',str(r/'benchmark.py'),'--port','8002','--label',label,'--output',str(r/'results'/f'{label}.json')]
 if only:cmd+=['--bench-only']
 subprocess.run(cmd,check=True)
def stop():
 s=json.loads((r/'active.json').read_text());os.kill(s['guard_pid'],signal.SIGTERM)
 for _ in range(50):
  if (Path(s['run'])/'exit.json').exists():return
  time.sleep(1)
 raise RuntimeError('native server did not stop')
# First benchmark was started separately. Do not interrupt an active request.
for _ in range(300):
 p=r/'results/plain.json'
 if p.exists() and json.loads(p.read_text()).get('finished'):break
 time.sleep(1)
else:raise RuntimeError('initial benchmark did not finish')
assert json.loads(p.read_text())['passed']
bench('plain-repeat',True)
for mode in ['mtp','dflash']:
 stop()
 with (r/f'{mode}-launch.log').open('w') as log:
  subprocess.Popen(['python3',str(r/'run-server.py'),'--mode',mode],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(2)
 subprocess.run(['python3',str(r/'wait-ready.py')],check=True)
 bench(mode)
 bench(mode+'-repeat',True)
(r/'comparison-complete.json').write_text(json.dumps({'finished':time.time(),'active':'dflash'})+'\n')
