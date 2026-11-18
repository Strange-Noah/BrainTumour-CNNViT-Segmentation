"""Bounded subprocess monitoring; no secret-bearing environment logging."""
import json, subprocess, time
from pathlib import Path
import psutil

def run(command,out,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'process.log').open('w') as stderr, (out/'system.jsonl').open('w') as log:
        process=subprocess.Popen(command,stdout=stderr,stderr=stderr,start_new_session=True);root=psutil.Process(process.pid);start=time.monotonic();reason=None;rss_history=[]
        while process.poll() is None:
            try:descendants=root.children(recursive=True)
            except psutil.NoSuchProcess:break
            procs=[root]+descendants;rss=[]
            for p in procs:
                try:
                    stat=Path(f'/proc/{p.pid}/stat').read_text().split(') ',1)[1].split()
                    rss.append(dict(pid=p.pid,rss=p.memory_info().rss,pss=getattr(p.memory_full_info(),'pss',None),read_bytes=p.io_counters().read_bytes,minor_faults=int(stat[7]),major_faults=int(stat[9])))
                except (psutil.NoSuchProcess,psutil.AccessDenied,AttributeError):pass
            vm=psutil.virtual_memory();swap=psutil.swap_memory();total=sum(p['rss'] for p in rss);rss_history.append(total);rss_history=rss_history[-120:]
            try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
            except Exception:gpu=[]
            log.write(json.dumps(dict(wall=time.time(),elapsed=time.monotonic()-start,rss=total,processes=rss,available=vm.available,total_ram=vm.total,swap_in=swap.sin,swap_out=swap.sout,gpu=gpu))+'\n');log.flush()
            if vm.available/vm.total<.15:reason='low_available_ram'
            if time.time()>deadline:reason='campaign_deadline'
            # Growth is reported; only actual RAM pressure or the hard deadline stops execution.
            # Mapping a new visit can legitimately increase summed RSS without exhausting RAM.
            if reason:
                import os,signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(2)
        code=process.wait();(out/'completion.json').write_text(json.dumps(dict(returncode=code,stop_reason=reason,seconds=time.monotonic()-start)))
        return code
