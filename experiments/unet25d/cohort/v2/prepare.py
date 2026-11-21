"""CPU-only auxiliary inventory + bounded real-data audit; no original MRI preparation."""
import gzip,io,json,os,tempfile,hashlib,time
from pathlib import Path
import numpy as np
from cloud.storage import connect,put_once,verify
from ..settings import keybase,read_json_record
from .settings import config,science_hash,prefix,aux_key,validate_catalogue
from .inventory import build
from .geometry import crop_origin
from .sampling import build_plan,digest
from .loading import Cache


def main():
    import torch
    if torch.cuda.is_available():raise RuntimeError('Auxiliary preparation is CPU only')
    from .tests import run_tests
    tests=run_tests();c=config();client,bucket=connect(output=True)
    if bucket!='novo-nordisk-pbl-outputs':raise ValueError('Private outputs only')
    ready=read_json_record(client,bucket,keybase()+'/ready.json');cat_receipt=ready['manifest']
    with tempfile.TemporaryDirectory(prefix='novo-private-v2-') as folder:
        root=Path(folder);root.chmod(0o700);catpath=root/'catalogue.json'
        with catpath.open('wb') as f:verify(client,bucket,cat_receipt['key'],cat_receipt['sha256'],f)
        cat=json.loads(catpath.read_text());validate_catalogue(cat);inventories={};reused=0
        for i,v in enumerate(cat['visits']):
            if v['partition']!='train':continue
            m=v['metadata'];e=m['exam_id'];record_key=prefix()+'/aux/visits/'+e+'.json'
            try:r=read_json_record(client,bucket,record_key);reused+=1
            except client.exceptions.NoSuchKey:
                path=root/'native.npy.gz'
                with path.open('wb') as f:verify(client,bucket,v['native_labels']['key'],v['native_labels']['sha256'],f)
                with gzip.open(path,'rb') as f:raw=np.load(f,allow_pickle=False)
                ids,components=build(raw,m);del raw
                comp_path=root/'components.npy.gz'
                with comp_path.open('wb') as f:
                    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as dest:np.save(dest,ids,allow_pickle=False)
                del ids
                receipt=put_once(client,bucket,prefix()+'/aux/components/'+hashlib.sha256(comp_path.read_bytes()).hexdigest()+'.npy.gz',comp_path)
                r=dict(exam=e,patient=m['patient_id'],partition='train',native_labels_sha256=v['native_labels']['sha256'],catalogue_sha256=cat_receipt['sha256'],science_sha256=science_hash(),components=components,labels=receipt,GT_FOV_preserved=True)
                p=root/'visit.json';p.write_text(json.dumps(r,sort_keys=True));put_once(client,bucket,record_key,p)
            if r['catalogue_sha256']!=cat_receipt['sha256'] or r['science_sha256']!=science_hash() or r['partition']!='train':raise ValueError('Auxiliary provenance mismatch')
            inventories[e]=r
            if len(inventories)%50==0:print(json.dumps(dict(event='v2_aux_progress',training_visits=len(inventories),gpu_allocated=False)),flush=True)
        train=[v for v in cat['visits'] if v['partition']=='train']
        if len(inventories)!=746:raise ValueError('Incomplete training lesion inventory')
        plan,_,audit=build_plan(train,inventories,c)
        for row in plan:
            if row['kind']!='replay':continue
            comp=next(x for x in inventories[row['exam']]['components'] if x['id']==row['component'])
            row['requested_scale']=row['scale'];row['origin'],row['scale'],row['partial_large_lesion']=crop_origin(comp,row['z'],row['scale'],c['crop_jitter_fraction'],next(v['metadata']['shape'] for v in train if v['metadata']['exam_id']==row['exam']),row['seed'])
        from collections import Counter
        audit['actual_replay_scales']=dict(Counter(row['scale'] for row in plan if row['kind']=='replay'));audit['plan_SHA']=digest(plan)
        def fetch(key,sha,path):
            with Path(path).open('wb') as f:verify(client,bucket,key,sha,f)
        cache=Cache(root/'bounded-real',fetch,c,inventories);sample_rows=[]
        # Train-only, small deterministic real audit: small/medium/large lesions,
        # local/full scales and both true negative categories. No test or new MRI preparation.
        replay=[row for row in plan if row['kind']=='replay'];chosen=[]
        for scale in (128,256,512):
            row=next(r for r in replay if r['scale']==scale);chosen.append(row)
        for category in ('negative_enhancing','negative_empty'):
            chosen.append(next(r for r in plan if r['negative_category']==category))
        visits={v['metadata']['exam_id']:v for v in train}
        for row in chosen:
            x,y,m,w,identity=cache.sample_v2(visits[row['exam']],row)
            from .loss import balanced_loss
            logits=torch.zeros_like(y[None],requires_grad=True);loss=balanced_loss(logits,y[None],m[None],w[None]);loss.backward()
            if torch.any(logits.grad[~m[None].bool()]!=0) or not torch.isfinite(loss):raise ValueError('Real masked loss gradient')
            if bool(y.any()) and not bool((logits.grad[y[None].bool()]<0).all()):raise ValueError('Real foreground gradient')
            if bool((~y.bool()&m.bool()).any()) and not bool((logits.grad[0][~y.bool()&m.bool()]>0).all()):raise ValueError('Real negative learning gradient')
            sample_rows.append(dict(positive=bool(y.any()),scale=row['scale'],target_pixels=int(y.sum()),valid_pixels=int(m.sum()),loss=float(loss.detach()),fallback=identity['mask_fallback']))
        invariant=dict(complete=True,science_sha256=science_hash(),catalogue_sha256=cat_receipt['sha256'],training_patients=210,training_visits=746,lesions=sum(len(r['components']) for r in inventories.values()),GT_geometric_coverage='all training components checked',GT_support_preservation='per-sample checked fallback to FOV before input suppression/loss; applies to every component occurrence',test_accessed=False,original_preparation_repeated=False,tests=tests,bounded_real_examples=sample_rows,mask_counts=cache.mask_counts,reused_aux_visits=reused)
        payloads={'inventory.json':inventories,'plan.json':plan,'GATE0_ACCEPTANCE.json':invariant,'SAMPLING_AUDIT.json':audit,'MASK_AND_LOSS_ACCEPTANCE.json':dict(verified=True,real_examples=sample_rows,loss_coefficients=c['loss_coefficients'],component_weight='bounded inverse-volume preference normalized per component and positive ROI',GT_loss_exclusion=0)}
        receipts={}
        for name,data in payloads.items():
            path=root/name;path.write_text(json.dumps(data,sort_keys=True));receipts[name]=put_once(client,bucket,prefix()+'/aux/'+name,path)
        record=dict(complete=True,science_sha256=science_hash(),source_sha=os.environ['NOVO_SOURCE_SHA'],catalogue=cat_receipt,objects=receipts)
        path=root/'ready.json';path.write_text(json.dumps(record,sort_keys=True));put_once(client,bucket,aux_key(),path)
        print(json.dumps(dict(event='v2_gate0_complete',gate0=invariant,sampling=audit,aux_receipt_key=aux_key())),flush=True)
        return record

if __name__=='__main__':main()
