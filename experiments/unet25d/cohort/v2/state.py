"""Strict v2 checkpoint identity, independent diagnostic and committed coverage."""
from .settings import SCHEMA,science_hash

def initial(catalogue_sha,inventory_sha,plan_sha):
    return dict(data_mode=SCHEMA,science_sha256=science_hash(),catalogue_sha256=catalogue_sha,inventory_sha256=inventory_sha,plan_sha256=plan_sha,step=0,cursor=0,mandatory_unique=0,replay_consumed=0,positive_samples=0,negative_samples=0,consumed_seconds=0.,phase='PHASE1',pending_validation=False,validation_progress=[],learning_gate_passed=False,early_diagnostic_complete=False,exposures={},difficulty={},scientific_stop=None)

def require_resume(old,cat,inv,plan):
    from ..settings import ROOT
    if old.get('data_mode')!=SCHEMA or old.get('science_sha256')!=science_hash() or old.get('catalogue_sha256')!=cat or old.get('inventory_sha256')!=inv or old.get('plan_sha256')!=plan:raise ValueError('v2 cannot load v1/other experiment checkpoint or optimizer')
    if old.get('normalization_migration') or old['cursor']!=2*old['step'] or old['mandatory_unique']+old['replay_consumed']!=old['cursor']:raise ValueError('Invalid v2 sampler/normalization state')
    if old.get('runtime_precision')!={'dtype':'float32','autocast':False,'grad_scaler':False,'tf32':False}:raise ValueError('Frozen FP32 checkpoint')
    return True
