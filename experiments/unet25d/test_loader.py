"""Distribution, deterministic resume and bounded file ownership checks."""
import tempfile,unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .data import SliceCache,sample_draws,locality_draws

def records():
 return [dict(exam_id=f'v{i}',patient_id=f'p{i//2}',positive=[2,3] if i<4 else [],negative=[0,1,4]) for i in range(8)]

def fixture(root,size=16,z=5):
 for i,r in enumerate(records()):
  p=root/r['exam_id'];p.mkdir();a=np.lib.format.open_memmap(p/'images.npy',mode='w+',dtype='float32',shape=(4,size,size,z));a[:]=i+np.arange(z)[None,None,None,:]/10;a.flush();del a
  y=np.zeros((size,size,z),np.uint8)
  if r['positive']:y[0,0,r['positive']]=1
  np.save(p/'target.npy',y)

class LocalityContract(unittest.TestCase):
 def test_exact_per_rank_multisets_and_resume_prefixes(self):
  rs=records()
  for world in (1,2):
   raw=sample_draws(rs,world*64*3,42,world);ordered=locality_draws(rs,len(raw),42,world)
   for start in range(0,len(raw),world*64):
    for rank in range(world):self.assertEqual(Counter(raw[start+rank:start+world*64:world]),Counter(ordered[start+rank:start+world*64:world]))
   for count in (1,63,64,65,127,128,129):self.assertEqual(locality_draws(rs,count,42,world),ordered[:count])
   for offset in (1,63,64,65):
    full=locality_draws(rs,world*(offset+128),42,world);short=locality_draws(rs,world*(offset+32),42,world)
    self.assertEqual(full[world*offset:world*(offset+32)],short[world*offset:])
 def test_tensor_values_bounds_and_no_reopen_within_visit(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);fixture(root);draws=[('v0',2),('v0',3),('v1',0)]
   size=sum((root/'v0'/n).stat().st_size for n in ['images.npy','target.npy']);ds=SliceCache(root,draws,2,size*2-1)
   x,y,_=ds[0];self.assertTrue(torch.allclose(x[0],torch.full_like(x[0],.1)));self.assertEqual(y.sum(),1)
   ds[1];self.assertEqual(ds.misses,1);ds[2];self.assertEqual(len(ds.cache),1);self.assertLessEqual(ds.mapped_bytes,size*2-1);self.assertTrue(torch.isfinite(x).all());ds.close();self.assertEqual(ds.mapped_bytes,0)
   with self.assertRaises(MemoryError):SliceCache(root,draws,2,size-1)[0]
 def test_persistent_worker_reuses_process_and_returns_same_samples(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);fixture(root);draws=locality_draws(records(),64,42);ds=SliceCache(root,draws,2,1024**2)
   loader=torch.utils.data.DataLoader(ds,batch_size=8,num_workers=1,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
   first=[x[:,0,0,0].tolist() for x,y,info in loader];pid=loader._iterator._workers[0].pid
   second=[x[:,0,0,0].tolist() for x,y,info in loader];self.assertEqual(first,second);self.assertEqual(pid,loader._iterator._workers[0].pid)
   del loader;ds.close()
if __name__=='__main__':unittest.main()
