"""Aggregate measured telemetry only; unavailable fields remain explicit."""
import json,sys
from pathlib import Path
import numpy as np

def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l]
def main():
    root=Path(sys.argv[1]);table=[]
    for folder in sorted(root.iterdir()):
        paths=list(folder.glob('rank-*.jsonl')) if folder.is_dir() else []
        if not paths:continue
        ranks=[read(p) for p in sorted(paths)];steady=[[r for r in rows if r['phase']=='steady'] for rows in ranks]
        if not all(steady):continue
        duration=max(rows[-1]['elapsed']-rows[0]['elapsed']+rows[0]['iteration_seconds'] for rows in steady);samples=sum(sum(r['samples'] for r in rows) for rows in steady);speed=samples/duration
        systems=read(folder/'system.jsonl');rss=[r['rss'] for r in systems];early=[r for r in steady[0][:max(1,len(steady[0])//3)]];late=steady[0][-max(1,len(steady[0])//3):]
        rate=lambda rows:sum(r['samples'] for r in rows)/sum(r['iteration_seconds'] for r in rows)
        row=dict(configuration=folder.name,local_batch=steady[0][0]['samples'],world_size=len(ranks),samples_per_second=speed,steady_seconds=duration,peak_vram_allocated_per_rank=[max(r['vram_allocated'] for r in rows) for rows in ranks],peak_vram_reserved_per_rank=[max(r['vram_reserved'] for r in rows) for rows in ranks],peak_rss=sum([]) if not rss else max(rss),early_rank0_throughput=rate(early),late_rank0_throughput=rate(late),data_wait_fraction=sum(r['data_wait_seconds'] for rows in steady for r in rows)/sum(r['iteration_seconds'] for rows in steady for r in rows),sustained_accepted=duration>=600,rss_plateau=bool(len(rss)>=30 and np.median(rss[-max(1,len(rss)//3):])<=1.10*np.median(rss[len(rss)//3:2*len(rss)//3])),late_to_middle_rss_ratio=float(np.median(rss[-max(1,len(rss)//3):])/max(1,np.median(rss[len(rss)//3:2*len(rss)//3]))) if len(rss)>=30 else None,sync_overhead='backward includes DDP synchronization; cannot isolate overlap from this timer',rss_limitation='sum RSS double-counts shared mmap pages; available RAM and page faults must also be reviewed')
        row['precision']=steady[0][0].get('precision','legacy-mixed')
        row['min_available_ram_fraction']=min((r['available']/r['total_ram'] for r in systems),default=None)
        row['swap_in_delta_bytes']=max((r['swap_in'] for r in systems),default=0)-min((r['swap_in'] for r in systems),default=0)
        hits=[max(max(r['cache']['hit']) for r in rows) for rows in ranks]
        misses=[max(max(r['cache']['miss']) for r in rows) for rows in ranks]
        row['cache_hit_fraction']=sum(hits)/max(1,sum(hits)+sum(misses))
        row['mmap_visit_opens_per_rank']=misses
        row['peak_mapped_bytes_per_rank']=[max(max(r['cache'].get('mapped_bytes',[0])) for r in rows) for rows in ranks]
        seconds=sum(r['iteration_seconds'] for rows in steady for r in rows)
        row['time_fraction']={key:sum(r[key] for rows in steady for r in rows)/seconds for key in ['data_wait_seconds','h2d_seconds','forward_seconds','backward_including_sync_seconds','optimizer_seconds']}
        table.append(row)
    baseline=next((r['samples_per_second'] for r in table if r['configuration']=='A-sustained'),None)
    for r in table:r['speedup_vs_A_sustained']=r['samples_per_second']/baseline if baseline else None
    (root/'performance.json').write_text(json.dumps(table,indent=2))
    if table:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for folder in root.iterdir():
            if not (folder/'system.jsonl').exists():continue
            rows=read(folder/'system.jsonl');fig,ax=plt.subplots();ax.plot([r['elapsed'] for r in rows],[r['rss']/1024**3 for r in rows]);ax.set(xlabel='Elapsed seconds',ylabel='Summed process RSS (GiB)',title=folder.name);fig.savefig(folder/'rss.png');plt.close(fig)
            if (folder/'rank-0.jsonl').exists():
                steps=read(folder/'rank-0.jsonl');fig,axes=plt.subplots(3,1,figsize=(8,9));times=[r['elapsed'] for r in steps];axes[0].plot(times,[r['samples']/r['iteration_seconds'] for r in steps]);axes[0].set_ylabel('Rank 0 samples/s');axes[1].plot(times,[100*r['data_wait_seconds']/r['iteration_seconds'] for r in steps]);axes[1].set_ylabel('Data wait %');axes[2].plot(times,[max(r['cache']['occupancy']) for r in steps]);axes[2].set(ylabel='Worker cache mappings',xlabel='Seconds');fig.tight_layout();fig.savefig(folder/'pipeline.png');plt.close(fig)
            fig,ax=plt.subplots()
            for gpu in range(2):
                pairs=[]
                for r in rows:
                    for line in r['gpu']:
                        fields=line.split(',')
                        if int(fields[0])==gpu:
                            try:pairs.append((r['elapsed'],float(fields[2])))
                            except ValueError:pass
                if pairs:ax.plot(*zip(*pairs),label=f'GPU {gpu}')
            ax.set(xlabel='Seconds',ylabel='GPU utilization %');ax.legend();fig.savefig(folder/'gpu.png');plt.close(fig)
    print(json.dumps(dict(configurations=len(table),no_measurements=not bool(table))))
if __name__=='__main__':main()
