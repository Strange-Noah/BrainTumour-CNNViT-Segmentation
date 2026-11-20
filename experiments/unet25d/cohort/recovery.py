"""Explicit, single-checkpoint normalization-policy migration; weights/Adam/cursor preserved."""
import hashlib,json
LEGACY_SCIENCE='c3a7a581affc95bb5ea8723c9d286c7f95376717de40bbecdb03e2887bf427b0'
CHECKPOINT_SHA='5db7514820877a5290e2dcb7ccb328298d96f0aa8b411011fef6de1182f99af4'
CHECKPOINT_STEP=74334

def legacy_config(science):
    c=dict(science);c['encoder_bn']='frozen-running-statistics-trainable-affine';c.pop('normalization_recovery')
    if hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()!=LEGACY_SCIENCE:raise ValueError('Unexpected migration science')
    return c

def require_migration(old,science,checkpoint_sha):
    if old['config']!=legacy_config(science) or old['step']!=CHECKPOINT_STEP or old['phase1_unique']!=148668 or old['source_sha']!='e818635a9e7df4d91792afaad7fc9ff3dd4b2a23' or checkpoint_sha!=CHECKPOINT_SHA:raise ValueError('Unapproved checkpoint migration')

def reset_statistics(model):
    import torch
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.reset_running_stats()

def enable_encoder_statistics(model):
    # Model.train() historically freezes encoder statistics; override that ONLY
    # for this explicitly versioned full-cohort normalization recovery.
    model.encoder.train(model.training)
    return model
