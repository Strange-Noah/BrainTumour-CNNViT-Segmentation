"""Small synthetic CPU comparison; no clinical data, preprocessing or GPU."""
import json,tempfile,time,platform
from pathlib import Path
from .data import SliceCache,sample_draws,locality_draws
from .test_loader import records,fixture

def main():
 result={'synthetic':True,'shape':[4,128,128,32],'samples':1024,'platform':platform.system(),'gpu_allocated':False,'limitations':'Warm local filesystem microprofile; not real MRI or Kaggle I/O throughput.'}
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);fixture(root,128,32)
  for name,draws,capacity in [('previous_random',sample_draws(records(),1024,42),1),('bounded_locality',locality_draws(records(),1024,42),2)]:
   ds=SliceCache(root,draws,capacity,32*1024**2);start=time.perf_counter();peak=0
   for i in range(len(ds)):
    x,y,info=ds[i];peak=max(peak,info['mapped_bytes'])
   seconds=time.perf_counter()-start
   result[name]=dict(seconds=seconds,samples_per_second=len(ds)/seconds,mmap_visit_opens=ds.misses,hits=ds.hits,peak_mapped_bytes=peak);ds.close()
 result['mmap_open_reduction_fraction']=1-result['bounded_locality']['mmap_visit_opens']/result['previous_random']['mmap_visit_opens']
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
