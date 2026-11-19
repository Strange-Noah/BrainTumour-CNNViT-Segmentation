"""GET-only private data lease; never place the lease in a checkpoint or public output."""
import hashlib,json,os,tempfile,time,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
import torch
from .loading import ShardCache
from .sampling import epoch_plan,resume_rank

class SignedFetch:
    def __init__(self,lease):self.lease=lease
    def __call__(self,key,expected,destination):
        lease=self.lease
        if time.time()>=lease['expires_at']:raise RuntimeError('Private input lease expired; refresh input on resume without re-preparation')
        parsed=urlsplit(lease['objects'][key]['url'])
        if parsed.scheme!='https' or not parsed.hostname or not parsed.hostname.endswith('.r2.cloudflarestorage.com'):raise ValueError('Unexpected private input host')
        destination=Path(destination);tmp=destination.with_name(destination.name+'.partial-'+str(os.getpid()))
        try:
            h=hashlib.sha256()
            with urllib.request.urlopen(lease['objects'][key]['url'],timeout=120) as src,tmp.open('wb') as out:
                while chunk:=src.read(1024**2):h.update(chunk);out.write(chunk)
            if h.hexdigest()!=expected:raise ValueError('Private object checksum mismatch')
            tmp.replace(destination)
        except Exception:tmp.unlink(missing_ok=True);raise RuntimeError('Private shard retrieval failed') from None

class TrainingInput(torch.utils.data.Dataset):
    def __init__(self,catalogue,lease,cache_root,epoch,cursor=0,rank=0,world=2,plan=None,mining=False):
        visits=[v for v in catalogue['visits'] if v['partition']=='train'];self.visits={v['metadata']['exam_id']:v for v in visits}
        from .settings import CONFIG
        config=json.loads(CONFIG.read_text())
        from .sampling import plan_digest
        self.plan,self.plan_sha=epoch_plan(visits,epoch,seed=config['seed'],samples_per_patient=config['samples_per_training_patient_per_coverage_epoch']) if plan is None else (plan,plan_digest(plan));self.draws=resume_rank(self.plan,cursor,rank,world);self.epoch=epoch;self.cursor=cursor
        self.mining=mining;self.properties={};self.property_root=Path(cache_root)/str(rank)/'gt-properties';self.fetch=SignedFetch(lease)
        self.cache=ShardCache(Path(cache_root)/str(rank),SignedFetch(lease),config['gpu_cache_bytes'],config['decoded_cache_bytes_per_worker'])
    def __len__(self):return len(self.draws)
    def __getitem__(self,index):
        visit,tile,z,category=self.draws[index];x,y=self.cache.sample(self.visits[visit],tile,z)
        if not self.mining:return x,y
        from .mining import lesion_properties
        if bool(y.any()) and visit not in self.properties:self.properties[visit]=lesion_properties(self.visits[visit],self.fetch,self.property_root)
        volume=self.properties.get(visit,{}).get(str((tile[0],tile[1],z)),0.)
        if bool(y.any()) and volume<=0:raise ValueError('Missing 3D lesion property')
        return x,y,dict(exam=visit,tile=list(tile),z=z,category=category,min_volume_mm3=volume)

def collate_mining(rows):
    if len(rows)!=1:raise ValueError('Frozen local batch 1')
    x,y,identity=rows[0];return x[None],y[None],identity
    def state_dict(self,global_cursor):return dict(epoch=self.epoch,global_cursor=global_cursor,plan_sha256=self.plan_sha)
