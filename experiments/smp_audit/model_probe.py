"""Synthetic CPU inference only; downloads the official pinned encoder weights."""
import hashlib, importlib.metadata as md, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main():
    assert md.version("segmentation-models-pytorch") == "0.5.0"
    torch.manual_seed(42); torch.set_num_threads(2)
    revision='7a57b34f723329ff020b3f8bc41771163c519d0c'
    path=hf_hub_download('smp-hub/resnet34.imagenet','model.safetensors',revision=revision)
    state=load_file(path)
    model=smp.Unet(encoder_name='resnet34',encoder_weights='imagenet',in_channels=12,classes=1,activation=None).eval()
    expected=state['conv1.weight'][:,torch.arange(12)%3]*0.25
    assert torch.equal(model.encoder.conv1.weight,expected)
    retained=[k for k,v in model.encoder.state_dict().items() if k!='conv1.weight' and k in state and torch.equal(v,state[k])]
    assert all(torch.equal(v,state[k]) for k,v in model.encoder.state_dict().items() if k!='conv1.weight' and k in state)
    results=[]
    with torch.inference_mode():
        for h,w in [(64,64),(240,240),(256,256),(255,257),(512,512)]:
            x=torch.randn(1,12,h,w); start=time.perf_counter()
            try:
                features=model.encoder(x); y=model(x)
                results.append(dict(input=list(x.shape),output=list(y.shape),features=[list(f.shape) for f in features],seconds=time.perf_counter()-start,finite=bool(torch.isfinite(y).all())))
            except RuntimeError as e: results.append(dict(input=list(x.shape),error=str(e)))
    fn=smp.encoders.get_preprocessing_fn('resnet34','imagenet')
    try: fn(np.zeros((8,8,12),dtype=np.float32)); preprocessing12='accepted'
    except ValueError as e: preprocessing12=str(e)
    report=dict(source_sha='420ce84b0c2df0286fa9bb2bd1499eea625c9b33',checkpoint_revision=revision,checkpoint_sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),checkpoint_bytes=Path(path).stat().st_size,packages={n:md.version(n) for n in ['torch','torchvision','segmentation-models-pytorch','timm','huggingface-hub','safetensors','numpy','Pillow']},python=sys.version,parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),encoder_buffers_absent_from_checkpoint=[k for k in model.encoder.state_dict() if k not in state],retained_encoder_tensors=len(retained),adaptation_exact=True,normalization=smp.encoders.get_preprocessing_params('resnet34','imagenet'),preprocessing12=preprocessing12,runs=results,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rss_unit='bytes' if sys.platform=='darwin' else 'KiB')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
