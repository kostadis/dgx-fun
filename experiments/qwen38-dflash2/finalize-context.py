import json,os,signal,subprocess,time
from pathlib import Path
r=Path.home()/'qwen38-dflash-trial'
for _ in range(1800):
 if (r/'comparison-complete.json').exists():break
 time.sleep(1)
else:raise RuntimeError('comparison not completed; leaving current mode untouched')
for label in ['plain','plain-repeat','mtp','mtp-repeat','dflash','dflash-repeat']:
 assert json.loads((r/'results'/f'{label}.json').read_text())['passed'],label
s=json.loads((r/'active.json').read_text());assert s['mode']=='dflash'
mem=json.loads((Path(s['run'])/'memory.jsonl').read_text().splitlines()[-1]);assert mem['MemAvailable']>16*2**30
os.kill(s['guard_pid'],signal.SIGTERM)
for _ in range(50):
 if (Path(s['run'])/'exit.json').exists():break
 time.sleep(1)
else:raise RuntimeError('native server did not stop')
with (r/'dflash-32k-launch.log').open('w') as log:
 subprocess.Popen(['python3',str(r/'run-server.py'),'--mode','dflash','--context','32768'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
subprocess.run(['python3',str(r/'wait-ready.py')],check=True)
subprocess.run(['python3',str(r/'benchmark.py'),'--port','8002','--label','dflash-32k','--output',str(r/'results/dflash-32k.json')],check=True)
subprocess.run(['python3',str(r/'context-tests/test-context.py'),'--context','32768','--target','30000'],check=True)
(r/'final-validation-complete.json').write_text(json.dumps({'finished':time.time(),'active':'dflash','context':32768})+'\n')
