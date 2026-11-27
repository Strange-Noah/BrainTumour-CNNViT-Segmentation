"""V3-only safe resume with primitive validation metadata."""
import math

import numpy as np
import torch

from ...train import RUNTIME_PRECISION
from ..v2.state import initial as initial_v2
from .settings import SCHEMA, science_hash


def initial(catalogue_sha, inventory_sha, plan_sha):
    return dict(initial_v2(catalogue_sha, inventory_sha, plan_sha),
                data_mode=SCHEMA, science_sha256=science_hash())


def plain_state(value):
    if isinstance(value, np.generic):
        return plain_state(value.item())
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise ValueError('Checkpoint metadata keys must be strings')
        return {key: plain_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_state(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError('Nonfinite checkpoint metadata')
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError(f'Unsupported checkpoint metadata type: {type(value).__name__}')


def load_checkpoint(path, catalogue, inventory, plan, source_sha):
    state = torch.load(path, map_location='cpu', weights_only=True)
    identity = (SCHEMA, science_hash(), catalogue, inventory, plan, source_sha)
    keys = ('data_mode', 'science_sha256', 'catalogue_sha256',
            'inventory_sha256', 'plan_sha256', 'source_sha')
    if tuple(state.get(k) for k in keys) != identity:
        raise ValueError('V3 resume requires matching source, science and inputs')
    if (state.get('runtime_precision') != RUNTIME_PRECISION
            or state.get('normalization_migration')
            or state['cursor'] != state['step'] * 2
            or state['cursor'] != state['mandatory_unique'] + state['replay_consumed']
            or len(state['rank_rng']) != 2):
        raise ValueError('Invalid V3 resume state')
    return state


def complete_phase1(state, config, plan_length):
    mandatory = config['mandatory_positions']
    divisor = config['replay_divisor']
    replay = (mandatory + divisor - 1) // divisor
    replay += (-(mandatory + replay)) % config['global_batch']
    if (
        plan_length != mandatory + replay
        or state['cursor'] != plan_length
        or state['step'] * config['global_batch'] != plan_length
        or state['mandatory_unique'] != mandatory
        or state['replay_consumed'] != replay
        or not state['learning_gate_passed']
        or state['pending_validation']
        or state['validation_progress']
    ):
        raise ValueError('Cannot complete Phase 1 without exact committed coverage')
    status = 'PHASE1_TRAINING_COMPLETE'
    state.update(phase=status, scientific_stop=status)
    return dict(
        status=status, training_complete=True, validation_performed=False,
        test_evaluated=False, phase2_launched=False,
        step=state['step'], cursor=state['cursor'],
        mandatory_unique=mandatory, replay_consumed=replay,
        science_sha256=state['science_sha256'],
    )
