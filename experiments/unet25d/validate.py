"""Complete frozen-validation visit, native T1c export and shared metrics."""
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from .train import Model,RUNTIME_PRECISION
from .data import SliceCache
from .export_native import export_native
from research.evaluate import evaluate

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
        prediction.flush();dataset.close()
        original,reference=export_native(prediction,meta,folder/'prediction',state['git_sha'],str(checkpoint.name))
        truth=np.load(cache/meta['exam_id']/'native-target.npy',mmap_mode='r')
        metrics=evaluate(original,truth,reference.affine);(folder/'metrics.json').write_text(json.dumps(metrics))
        del prediction,dataset,original,truth
if __name__=='__main__':main()
