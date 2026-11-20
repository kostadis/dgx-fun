import argparse,datetime,json,os,signal,subprocess,time,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['plain','mtp','dflash'],required=True);p.add_argument('--context',type=int,default=8192);p.add_argument('--port',type=int,default=8002);p.add_argument('--output-tokens',type=int,default=32000);a=p.parse_args()
r=Path.home()/'qwen38-dflash-trial'
assert (r/'download-complete.json').exists() and (r/'build-complete.txt').exists()
running=subprocess.check_output(['docker','ps','--format','{{.Names}}']).decode().splitlines();assert 'qwen38-flash' not in running,'Stop Qwen vLLM before starting this trial'
run=r/'runs'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+a.mode);run.mkdir(parents=True)
unit='qwen38-dflash-'+uuid.uuid4().hex[:10]+'.scope'
binary=r/'source/build/bin/llama-server';model=r/'model/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00002.gguf'
cmd=[str(binary),'-m',str(model),'-ngl','99','-c',str(a.context),'-n',str(a.output_tokens),'-b','1024','-ub','1024','-np','1','-fa','on','--fit','off','--load-mode','mmap','--cache-ram','0','--no-cache-prompt','--no-context-shift','--host','0.0.0.0','--port',str(a.port),'--alias','qwen3.8-flash-next-gguf','--jinja','--reasoning-format','deepseek','--chat-template-kwargs','{"enable_thinking":false}','--temp','0','--top-p','1','--top-k','0','--min-p','0','--seed','604','-t','10','-tb','20']
if a.mode=='plain':cmd+=['--spec-type','none']
else:
 sidecar='mtp-Qwen3.8-Flash-Next-Q4_0.gguf' if a.mode=='mtp' else 'dflash-Qwen3.8-Flash-Next-Q4_0.gguf'
 cmd+=['-md',str(r/'model'/sidecar),'-ngld','99','--spec-type','draft-mtp' if a.mode=='mtp' else 'draft-dflash','--spec-draft-n-max','2' if a.mode=='mtp' else '5','--spec-draft-p-min','0.75']
full=['systemd-run','--user','--scope','--quiet','--unit='+unit,'-p','MemoryMax=107374182400','-p','MemorySwapMax=0',*cmd]
state={'time':time.time(),'mode':a.mode,'context':a.context,'output_tokens':a.output_tokens,'port':a.port,'run':str(run),'unit':unit,'command':cmd,'guard_pid':os.getpid()}
(run/'launch.json').write_text(json.dumps(state,indent=2)+'\n');(r/'active.json').write_text(json.dumps(state,indent=2)+'\n')
signals=[]
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda n,f:signals.append(n))
failure=None
with (run/'server.log').open('w') as log,(run/'memory.jsonl').open('w',buffering=1) as memlog:
 child=subprocess.Popen(full,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state['child_pid']=child.pid;(r/'active.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  while child.poll() is None:
   values={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','SwapFree:'))}
   memlog.write(json.dumps({'time':time.time(),**values})+'\n')
   if values['MemAvailable']<8*1024**3:failure='MemAvailable below 8 GiB';break
   if signals:break
   time.sleep(1)
 finally:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 state.update(returncode=child.returncode,failure=failure,signals=signals,finished=time.time());(run/'exit.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'run':str(run),'returncode':child.returncode,'failure':failure}),flush=True)
raise SystemExit(1 if failure else 0 if signals else child.returncode)
