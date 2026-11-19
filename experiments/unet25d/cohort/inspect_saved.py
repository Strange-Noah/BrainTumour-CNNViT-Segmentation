"""One private CPU diagnostic from saved prediction/GT; never run a model."""
import hashlib,io,json,os,tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
from cloud.storage import connect,verify,put_once
from ..persist_cache import read,sha
from ..repair_validation import CACHE_PREFIX,CACHE_SHA

def main():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c,b=connect(output=True);raw=read(c,CACHE_PREFIX+'/manifest.json')
    if sha(raw)!=CACHE_SHA:raise ValueError('Original cache metadata identity mismatch')
    files={r['path']:r for r in json.loads(raw)['files']}
    def cache_file(name):
        row=files[name];data=b''
        for chunk in row['chunks']:
            block=read(c,CACHE_PREFIX+'/chunks/'+chunk['sha256'])
            if len(block)!=chunk['bytes'] or sha(block)!=chunk['sha256']:raise ValueError('Private cache integrity failure')
            data+=block
        if sha(data)!=row['sha256']:raise ValueError('Private file integrity failure')
        return data
    meta=json.loads(cache_file('prepared/validation-index.json'))[0]
    key='novo-nordisk/experiments/unet25d/bridge/44292134002/cpu-recovery/44333112852/report.json';raw=read(c,key)
    if sha(raw)!='b4bf5e24d76622ac84fa214a3177f6acefe6fd9391a816897f97fdfbb92d87d1':raise ValueError('Saved prediction receipt mismatch')
    row=next(r for r in json.loads(raw)['outputs'] if r['key'].endswith('/seg.nii.gz'))
    with tempfile.TemporaryDirectory(prefix='novo-private-inspect-') as d:
        root=Path(d);root.chmod(0o700);path=root/'prediction.nii.gz'
        with path.open('wb') as f:verify(c,b,row['key'],row['sha256'],f)
        image=nib.load(path);pred=np.asarray(image.dataobj).astype(bool);truth=np.load(io.BytesIO(cache_file('prepared/'+meta['exam_id']+'/native-target.npy')),allow_pickle=False).astype(bool)
        from research.dataset import Dataset
        from .prepare import ROOT
        manifest=root/'manifest.jsonl';manifest.write_bytes(cache_file('manifest.jsonl'));dataset=Dataset(manifest,ROOT/'protocol/v1/patients.csv',root/'raw','validation')
        visit=next(v for v in dataset.visits if v['exam_id']==meta['exam_id']);reference=nib.load(dataset.fetch(visit['modalities']['t1c']))
        aligned=pred.shape==truth.shape and np.allclose(image.affine,meta['native_affine'],atol=1e-4) and image.shape==reference.shape and np.allclose(image.affine,reference.affine,atol=1e-4)
        if not aligned:raise ValueError('Gross native-grid mismatch')
        tp=int((pred&truth).sum());p=int(pred.sum());t=int(truth.sum());center=np.array(pred.shape)//2
        if p or t:
            mask=truth if t else pred;z=int(np.argmax(mask.sum(axis=(0,1))));xy=np.round(np.argwhere(mask[:,:,z]).mean(0)).astype(int);center=np.array([xy[0],xy[1],z])
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for axis,ax in enumerate(axes):
            gt=np.take(truth,int(center[axis]),axis=axis);pr=np.take(pred,int(center[axis]),axis=axis)
            selection=[slice(None)]*3;selection[axis]=int(center[axis]);background=np.asarray(reference.dataobj[tuple(selection)],dtype=np.float32);values=background[background!=0];lo,hi=np.percentile(values,[1,99]) if values.size else (0,1)
            ax.imshow(np.rot90(background),cmap='gray',vmin=lo,vmax=hi)
            if gt.any():ax.contour(np.rot90(gt),levels=[.5],colors=['lime'],linewidths=.8)
            if pr.any():ax.contour(np.rot90(pr),levels=[.5],colors=['red'],linewidths=.8)
            ax.set_title(['Sagittal','Coronal','Axial'][axis]+' native overlay');ax.axis('off')
        fig.suptitle('Private diagnostic only: red prediction, green ground truth; original T1c background');fig.tight_layout();figure=root/'overlap.png';fig.savefig(figure,dpi=150);plt.close(fig)
        report=dict(gpu_allocated=False,new_inference=False,empty_prediction=p==0,empty_ground_truth=t==0,native_grid_matches=True,dice=2*tp/(p+t) if p+t else 1,prediction_false_positive_fraction=(p-tp)/p if p else None,ground_truth_recall=tp/t if t else None,interpretation='Native grid checked against original T1c. Private overlays support localization/boundary/false-positive review; header agreement alone does not establish anatomical registration or model quality.')
        prefix='novo-nordisk/experiments/unet25d/diagnostics/'+os.environ['GITHUB_RUN_ID'];report['private_figure']=put_once(c,b,prefix+'/overlap.png',figure);path=root/'report.json';path.write_text(json.dumps(report,indent=2));receipt=put_once(c,b,prefix+'/report.json',path)
        print(json.dumps(dict(gpu_allocated=False,new_inference=False,empty_prediction=report['empty_prediction'],native_grid_matches=True,private_report=receipt)))
if __name__=='__main__':main()
