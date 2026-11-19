"""The single normal finalization: receipt-derived coverage plus one private CPU input batch."""
import hashlib,json,re,tempfile
from pathlib import Path
from .settings import CONFIG,science_hash,keybase

def coverage(records,expected):
    from .tiling import windows
    import numpy as np
    if len(expected)!=905 or len({r['exam_id'] for r in expected})!=905:raise ValueError('Expected cohort identity mismatch')
    actual={r['metadata']['exam_id']:r for r in records}
    if len(actual)!=len(records) or set(actual)!={r['exam_id'] for r in expected}:raise ValueError('Missing or duplicate prepared visits')
    identities={};counts={p:0 for p in ['train','validation']};oblique=oversized=0
    for row in expected:
        r=actual[row['exam_id']];m=r['metadata'];counts[r['partition']]+=1
        if m['patient_id']!=row['patient_id'] or m['grid_policy']!='native-T1c-with-affine' or m['oblique_resampled'] or len(m['normalization'])!=4:raise ValueError('Frozen modality/geometry recipe mismatch')
        shape=m['shape'];depth=shape[2];shards=r['shards']
        if not shards or shards[0]['start']!=0 or shards[-1]['end']!=depth or any(a['end']!=b['start'] for a,b in zip(shards,shards[1:])):raise ValueError('Incomplete shard range')
        if [t['origin'] for t in m['tiles']]!=[list(t) for t in windows(shape)]:raise ValueError('Missing spatial tiles')
        for tile in m['tiles']:
            a,b=tile['positive'],tile['negative']
            if len(a)+len(b)!=depth or set(a)&set(b) or set(a)|set(b)!=set(range(depth)):raise ValueError('Incomplete raw-label sampling range')
        for shard in shards:
            lo=max(0,shard['start']-1);hi=min(depth,shard['end']+1)
            if not 0<shard['end']-shard['start']<=8 or shard['halo_start']!=lo or shard['decoded_bytes']!=(hi-lo)*4*shape[0]*shape[1]*4+(shard['end']-shard['start'])*shape[0]*shape[1]:raise ValueError('Shard channel/label/context identity mismatch')
        for obj in [*shards,r['native_labels']]:
            if not re.fullmatch('[0-9a-f]{64}',obj['sha256']) or obj['bytes']<=0 or not obj['key'].startswith(keybase()+'/') or obj['sha256'] not in obj['key']:raise ValueError('Invalid recorded object identity')
            identity=(obj['sha256'],obj['bytes'])
            if obj['key'] in identities and identities[obj['key']]!=identity:raise ValueError('Conflicting recorded shard')
            identities[obj['key']]=identity
        affine=np.asarray(m['native_affine']);canonical=np.asarray(m['canonical_affine'])
        if affine.shape!=(4,4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3]))<=0:raise ValueError('Invalid native physical geometry')
        # Closest-RAS orientation and recorded padding must preserve the actual oblique lattice.
        import nibabel as nib
        transform=np.asarray(m['orientation_transform']);shift=np.eye(4);shift[:3,3]=[-x[0] for x in m['padding']]
        if not np.allclose(canonical,affine@nib.orientations.inv_ornt_aff(transform,m['native_shape'])@shift,atol=1e-4):raise ValueError('Native affine preservation mismatch')
        oblique+=bool(m['source_oblique']);oversized+=max(shape[:2])>512
    if counts!={'train':746,'validation':159} or oblique!=51 or oversized!=4:raise ValueError('Frozen cohort/oversized/oblique completeness mismatch')
    return dict(visits=counts,shard_count=sum(len(r['shards']) for r in records),native_label_objects=len(records),compressed_shard_bytes=sum(s['bytes'] for r in records for s in r['shards']),compressed_native_label_bytes=sum(r['native_labels']['bytes'] for r in records),unique_prepared_objects=len(identities),oversized_visits=oversized,oblique_visits=oblique,missing_visits=0,duplicate_visits=0,incomplete_ranges=0,conflicting_objects=0,integrity_basis='SHA-verified immutable preparation receipts; each referenced data object GET-verified at preparation, no routine full data rehash')

def check_published_input(dataset,version,folder):
    from cloud.kaggle_benchmark_bridge import cli
    from .stream import TrainingInput
    import torch
    if torch.cuda.is_available():raise RuntimeError('Final input check is CPU only')
    with tempfile.TemporaryDirectory(prefix='novo-private-input-check-') as d:
        root=Path(d);root.chmod(0o700)
        for name in ['catalogue.json','private-data-lease.json','training-config.json']:
            cli('datasets','download','-d',dataset+'/'+str(version),'-f',name,'-p',str(root),'-q')
            if not (root/name).exists():
                import zipfile
                archive=root/(name+'.zip')
                with zipfile.ZipFile(archive) as z:(root/name).write_bytes(z.read(name))
            if (root/name).read_bytes()!=(Path(folder)/name).read_bytes():raise ValueError('Published input differs from original private package')
        catalogue=json.loads((root/'catalogue.json').read_text());lease=json.loads((root/'private-data-lease.json').read_text())
        data=TrainingInput(catalogue,lease,root/'cache',0)
        x,y=next(iter(torch.utils.data.DataLoader(data,batch_size=1,num_workers=0)))
        if list(x.shape)!=[1,12,512,512] or list(y.shape)!=[1,1,512,512] or not torch.isfinite(x).all() or not torch.isin(y,torch.tensor([0.,1.])).all():raise ValueError('Invalid actual training input batch')
        return dict(verified=True,cpu_only=True,new_inference=False,catalogue_sha256=hashlib.sha256((root/'catalogue.json').read_bytes()).hexdigest(),input_shape=list(x.shape),target_shape=list(y.shape),input_batch_sha256=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest(),science_sha256=science_hash(),dataset=dataset,version=version)
