"""Complete frozen-validation visit, native T1c export and shared metrics."""
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from .train import Model,RUNTIME_PRECISION
from .data import SliceCache,restore
from research.evaluate import evaluate,export

def main():
    cache,checkpoint,output=map(Path,sys.argv[1:]);output.mkdir(parents=True,exist_ok=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['status']!='success' or state.get('runtime_precision')!=RUNTIME_PRECISION:raise ValueError('Validation requires healthy FP32 checkpoint')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Model(encoder_name='resnet34',encoder_weights=None,in_channels=12,classes=1,activation=None);model.load_state_dict(state['model']);model.cuda().eval()
    for meta in json.loads((cache/'validation-index.json').read_text()):
        folder=output/meta['exam_id'];folder.mkdir(exist_ok=True);z_count=meta['shape'][2]
        prediction=np.lib.format.open_memmap(folder/'canonical-prediction.npy',mode='w+',dtype='float32',shape=tuple(meta['shape']))
        dataset=SliceCache(cache,[(meta['exam_id'],z) for z in range(z_count)])
        with torch.inference_mode():
            for z in range(z_count):
                x,_,_=dataset[z]
                logits=model(x[None].cuda())
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite validation logits')
                prediction[:,:,z]=logits[0,0].float().cpu().numpy()
        original=restore(prediction,meta,order=1)>=0;truth=np.load(cache/meta['exam_id']/'native-target.npy',mmap_mode='r')
        reference=nib.Nifti1Image(np.empty(meta['native_shape'],dtype=np.uint8),np.asarray(meta['native_affine']));reference.header.set_xyzt_units('mm')
        metrics=evaluate(original,truth,reference.affine);(folder/'metrics.json').write_text(json.dumps(metrics))
        export(original,reference,folder,dict(patient_id=meta['patient_id'],exam_id=meta['exam_id'],task='enhancing',git_sha=state.get('git_sha','unrecorded'),model_version='unet25d-benchmark',checkpoint_ref=str(checkpoint.name)))
        del prediction,dataset,original,truth
if __name__=='__main__':main()
