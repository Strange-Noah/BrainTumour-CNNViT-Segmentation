"""Bounded real-input corrected-Gate1 checks, without constructing a model."""
import json,tempfile
from pathlib import Path
import torch
from .settings import config,science_hash
from .train import fixed_rows
from .loading import Cache
from .resilient_fetch import SignedFetch
from .loss import balanced_loss

def run(folder):
    folder=Path(folder);c=config();cat=json.loads((folder/'catalogue.json').read_text());inv=json.loads((folder/'inventory.json').read_text());plan=json.loads((folder/'plan.json').read_text());lease=json.loads((folder/'private-data-lease.json').read_text());rows=fixed_rows(plan,inv,c);visits={v['metadata']['exam_id']:v for v in cat['visits'] if v['partition']=='train'};out=[]
    with tempfile.TemporaryDirectory() as d:
        cache=Cache(Path(d),SignedFetch(lease),c,inv)
        for row in rows:
            x,y,m,w,identity=cache.sample_v2(visits[row['exam']],row)
            z=torch.zeros_like(y[None],requires_grad=True);loss=balanced_loss(z,y[None],m[None],w[None]);loss.backward()
            if bool((z.grad[~m[None].bool()]!=0).any()) or not torch.isfinite(loss):raise ValueError('Corrected real loss/mask check')
            components=len(identity['gate_component_masks'])
            if row['gate_role']=='tiny_capacity' and components!=1:raise ValueError('Selected isolated tiny example contains incidental GT')
            if row['gate_role']=='tiny_capacity' and not bool((z.grad[y[None].bool()]<0).all()):raise ValueError('Tiny foreground gradient absent')
            out.append(dict(role=row['gate_role'],scale=row['scale'],target_pixels=int(y.sum()),selected_pixels=identity.get('selected_component_foreground_pixels'),components=components,valid_pixels=int(m.sum()),loss=float(loss.detach()),finite_input=bool(torch.isfinite(x).all())))
    result=dict(event='v2_corrected_real_cpu_acceptance',science_sha256=science_hash(),examples=out,gpu_allocated=False,plan_unchanged=True)
    (folder/'CORRECTED_CPU_ACCEPTANCE.json').write_text(json.dumps(result));print(json.dumps(result),flush=True);return result
