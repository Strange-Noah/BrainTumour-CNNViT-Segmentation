"""Measure the exact V3 loss and gradients on the V2 frozen capacity examples."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import torch

from ..v2.loading import Cache
from ..v2.loss import balanced_loss
from ..v2.resilient_fetch import SignedFetch
from ..v2.sampling import audit_plan, mandatory
from .inputs import FROZEN_FILES, PLAN_SHA, file_sha
from .loss import masked_loss
from .settings import config, science_hash, validate_catalogue
from .train import fixed_rows


def run(folder, examples=False):
    folder = Path(folder)
    for name, sha in FROZEN_FILES.items():
        if file_sha(folder / name) != sha:
            raise ValueError(f'Frozen {name} mismatch')
    c = config()
    cat = json.loads((folder / 'catalogue.json').read_text())
    validate_catalogue(cat)
    inv = json.loads((folder / 'inventory.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    train = {r['metadata']['exam_id']: r for r in cat['visits']
             if r['partition'] == 'train'}
    if set(train) != set(inv):
        raise ValueError('Train-only inventory identity')
    for exam, item in inv.items():
        row = train[exam]
        if (item['partition'] != 'train' or item['patient'] != row['metadata']['patient_id']
                or item['catalogue_sha256'] != FROZEN_FILES['catalogue.json']
                or item['native_labels_sha256'] != row['native_labels']['sha256']):
            raise ValueError('Inventory provenance')
    audit = audit_plan(plan, c)
    if audit['plan_SHA'] != PLAN_SHA:
        raise ValueError('Frozen pairing mismatch')
    identity = lambda r: (r['exam'], tuple(r['origin']), r['z'], r['positive'])
    if {identity(r) for r in mandatory(list(train.values()))} != {
            identity(r) for r in plan if r['kind'] == 'mandatory'}:
        raise ValueError('Mandatory set mismatch')
    replay = [r for r in plan if r['kind'] == 'replay']
    by_patient = Counter(r['patient'] for r in replay)
    by_visit = Counter(r['exam'] for r in replay)
    by_lesion = Counter((r['exam'], r['component']) for r in replay)
    for key, observed in [('patient', by_patient), ('visit', by_visit), ('lesion', by_lesion)]:
        if max(observed.values()) > c[f'replay_{key}_cap']:
            raise ValueError('Replay cap exceeded')
    report = dict(science_sha256=science_hash(), cohort=dict(
        train_patients=210, train_exams=len(train), val_patients=45,
        val_exams=len(cat['visits'])-len(train), test_included=False),
        audit=audit, replay_max=dict(patient=max(by_patient.values()),
        visit=max(by_visit.values()), lesion=max(by_lesion.values())),
        replay_scales=dict(Counter(r['scale'] for r in replay)), examples=[])
    if not examples:
        return report
    lease = json.loads((folder / 'private-data-lease.json').read_text())
    if lease['catalogue_sha256'] != FROZEN_FILES['catalogue.json']:
        raise ValueError('Wrong data lease')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        cache = Cache(d, SignedFetch(lease), c, inv)
        for row in fixed_rows(plan, inv, c):
            x, y, m, w, identity = cache.sample_v2(train[row['exam']], row)
            # Non-constant logits make the redistribution observable in BCE.
            z = torch.linspace(-1., 1., y.numel()).reshape_as(y[None]).requires_grad_()
            valid = m[None].bool()
            pos = y[None].bool()
            neg = valid & ~pos
            loss, parts = masked_loss(z, y[None], m[None], w[None], True)
            _, v2 = balanced_loss(z, y[None], m[None], w[None], True)
            element = torch.nn.functional.binary_cross_entropy_with_logits(
                z, y[None], reduction='none')
            prior = element[valid].mean()
            ref = ((element[pos] * w[None][pos]).sum() + element[neg].sum()) / valid.sum()
            torch.testing.assert_close(parts['bce'], ref)
            torch.testing.assert_close(w[None][pos].sum(), pos.sum().float())
            new_grad, v2_grad, prior_grad = [
                torch.autograd.grad(b, z, retain_graph=True)[0]
                for b in (parts['bce'], v2['bce'], prior)
            ]
            torch.testing.assert_close(new_grad[pos], prior_grad[pos]*w[None][pos])
            torch.testing.assert_close(new_grad[neg], prior_grad[neg])
            if bool(pos.any()) and bool(neg.any()):
                torch.testing.assert_close(v2_grad[pos], new_grad[pos]*valid.sum()/(2*pos.sum()))
                torch.testing.assert_close(v2_grad[neg], new_grad[neg]*valid.sum()/(2*neg.sum()))
            loss.backward()
            if (not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(z.grad).all())
                    or bool((z.grad[~valid] != 0).any())
                    or (bool(pos.any()) and not bool((z.grad[pos] < 0).all()))):
                raise ValueError('Real V3 mask/gradient contract')
            n = int(y.sum())
            components = len(identity['gate_component_masks'])
            if row['gate_role']=='tiny_capacity' and components != 1:
                raise ValueError('Frozen tiny gate crop is not isolated')
            report['examples'].append(dict(role=row['gate_role'], target_pixels=n,
                valid_pixels=int(m.sum()), components=components, bce=float(parts['bce'].detach()),
                previous_v3_bce=float(prior.detach()), v2_bce=float(v2['bce'].detach()),
                positive_coefficient=float(pos.sum()/valid.sum()),
                max_positive_bce_gradient=float(new_grad[pos].abs().max()) if n else None,
                previous_v3_max_positive_bce_gradient=float(prior_grad[pos].abs().max()) if n else None,
                v2_max_positive_bce_gradient=float(v2_grad[pos].abs().max()) if n else None,
                dice_loss=float(parts['soft_dice'].detach()),
                min_positive_total_gradient=float(z.grad[pos].min()) if n else None,
                max_negative_total_gradient=float(z.grad[valid & ~pos].max())))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    p.add_argument('--examples', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.folder, a.examples)))
