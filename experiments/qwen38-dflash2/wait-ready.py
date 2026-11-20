import json,os,time,urllib.request
from pathlib import Path
r=Path.home()/'qwen38-dflash-trial';start=time.monotonic()
for _ in range(600):
 state=json.loads((r/'active.json').read_text())
 try:os.kill(state['guard_pid'],0)
 except ProcessLookupError:raise RuntimeError('guard exited during startup')
 if (Path(state['run'])/'exit.json').exists():raise RuntimeError('server exited during startup')
 try:
  with urllib.request.urlopen('http://127.0.0.1:8002/health',timeout=5) as h:health=json.load(h)
  if health.get('status')=='ok':
   v={'time':time.time(),'startup_seconds':time.monotonic()-start,'health':health,'state':state}
   (Path(state['run'])/'ready.json').write_text(json.dumps(v,indent=2)+'\n');print('READY',state['mode'],round(v['startup_seconds'],2),state['run'],flush=True);break
 except (OSError,ValueError):pass
 time.sleep(2)
else:raise RuntimeError('server readiness timeout')
