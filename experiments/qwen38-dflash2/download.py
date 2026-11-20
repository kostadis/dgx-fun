import concurrent.futures,fcntl,hashlib,json,os,signal,subprocess,threading,time
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_url
root=Path.home()/'qwen38-dflash-trial';dest=root/'model';dest.mkdir(exist_ok=True)
lock=(root/'download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
children={};mutex=threading.Lock();stopping=threading.Event()
def stop(signum,frame):
    stopping.set()
    with mutex:
        for child in list(children.values()):
            if child.poll() is None:child.terminate()
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
repo='ggml-org/Qwen3.8-Flash-Next-GGUF';revision='91fb00353e766eb43944ed223665ea4fb87dad47'
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
names={'Qwen3.8-Flash-Next-IQ4_NL-00001-of-00002.gguf','Qwen3.8-Flash-Next-IQ4_NL-00002-of-00002.gguf','mtp-Qwen3.8-Flash-Next-Q4_0.gguf','dflash-Qwen3.8-Flash-Next-Q4_0.gguf'}
files=[f for f in info.siblings if f.rfilename in names]
assert len(files)==4
manifest={'repo':repo,'revision':info.sha,'files':[{'path':f.rfilename,'size':f.size,'sha256':f.lfs.sha256} for f in files]}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def fetch(f):
    if stopping.is_set():raise RuntimeError('download stopped')
    target=dest/Path(f.rfilename).name;partial=Path(str(target)+'.partial')
    candidate=target if target.exists() else partial
    if not target.exists() and (not partial.exists() or partial.stat().st_size<f.size):
        print('download',target.name,f.size,flush=True)
        url=hf_hub_url(repo,f.rfilename,revision=revision)
        with (root/(target.name+'.curl.log')).open('a') as log:
            child=subprocess.Popen(['curl','--fail','--location','--retry','5','--retry-delay','3','--continue-at','-','--output',str(partial),url],stdout=subprocess.DEVNULL,stderr=log)
            with mutex:
                children[target.name]=child
                if stopping.is_set():child.terminate()
            code=child.wait()
            with mutex:children.pop(target.name,None)
            if code:raise RuntimeError(f'{target.name}: curl exited {code}')
    if candidate.stat().st_size!=f.size:raise RuntimeError(f'{target.name}: size mismatch {candidate.stat().st_size} != {f.size}')
    h=hashlib.sha256()
    with candidate.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=f.lfs.sha256:raise RuntimeError(f'{target.name}: SHA256 mismatch')
    if candidate==partial:partial.rename(target)
    print('verified',target.name,flush=True)
    return target.name
errors=[];verified=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    jobs={pool.submit(fetch,f):f.rfilename for f in files}
    for job in concurrent.futures.as_completed(jobs):
        try:verified.append(job.result())
        except Exception as error:
            errors.append(str(error));print('ERROR',error,flush=True)
if errors:raise RuntimeError(errors)
(root/'download-complete.json').write_text(json.dumps({'revision':revision,'finished':time.time(),'verified_files':len(verified)},indent=2)+'\n')
print('ALL VERIFIED',len(verified),flush=True)
