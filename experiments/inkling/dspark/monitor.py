"""Bounded-memory host sampling; stop only the candidate below the RAM floor."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={'unit':a.unit,'started_unix':time.time(),'samples':0,'floor_tripped':False,
            'min_host_available_bytes':None,'max_swap_used_bytes':0,'max_process_rss_bytes':0}
    while subprocess.run(['systemctl','--user','is-active','--quiet',a.unit]).returncode==0:
        memory={line.split(':')[0]:int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines()}
        available=memory['MemAvailable'];result['samples']+=1
        result['min_host_available_bytes']=min(available,result['min_host_available_bytes'] or available)
        result['last_host_available_bytes']=available
        result['max_swap_used_bytes']=max(result['max_swap_used_bytes'],memory['SwapTotal']-memory['SwapFree'])
        pid=subprocess.check_output(['systemctl','--user','show',a.unit,'--value','--property=MainPID'],text=True).strip()
        try:
            status=Path('/proc')/pid/'status'
            rss=next(int(line.split()[1])*1024 for line in status.read_text().splitlines() if line.startswith('VmRSS:'))
            result['max_process_rss_bytes']=max(result['max_process_rss_bytes'],rss)
        except (FileNotFoundError,StopIteration):pass
        if available<12*1024**3:
            result['floor_tripped']=True
            subprocess.run(['systemctl','--user','stop',a.unit],check=False)
        result['updated_unix']=time.time()
        temporary=a.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(a.output)
        if result['floor_tripped']:break
        time.sleep(2)
    result['finished_unix']=time.time()
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
