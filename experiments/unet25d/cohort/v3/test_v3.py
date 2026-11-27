import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from ...train import RUNTIME_PRECISION
from ..v2.loss import balanced_loss
from ..v2.settings import SCHEMA as V2_SCHEMA, science_hash as v2_hash
from ..v2.validation import validate_patients
from .loss import masked_loss
from .persistence import verify_lease
from .settings import SCHEMA, prefix, science_hash
from .state import initial, load_checkpoint, plain_state
from . import train


class MaskedBCETests(unittest.TestCase):
    def check_case(self, target, mask):
        y = torch.tensor(target, dtype=torch.float32).view(1, 1, 1, -1)
        m = torch.tensor(mask, dtype=torch.bool).view_as(y)
        z = torch.linspace(-2, 1, y.numel()).view_as(y).requires_grad_()
        total, parts = masked_loss(z, y, m, return_parts=True)
        ref = F.binary_cross_entropy_with_logits(z[m], y[m])
        torch.testing.assert_close(parts['bce'], ref)
        _, old = balanced_loss(z, y, m, return_parts=True)
        torch.testing.assert_close(parts['soft_dice'], old['soft_dice'], rtol=0, atol=0)
        torch.testing.assert_close(total, .5 * ref + .5 * old['soft_dice'])
        self.assertTrue(torch.isfinite(total))
        grad = torch.autograd.grad(parts['bce'], z, retain_graph=True)[0]
        torch.testing.assert_close(grad, torch.autograd.grad(ref, z, retain_graph=True)[0])
        total.backward()
        self.assertEqual(int(torch.count_nonzero(z.grad[~m])), 0)
        self.assertTrue(bool((grad[m & y.bool()] < 0).all()))

    def test_all_negative(self):
        self.check_case([0]*20, [1]*20)

    def test_tiny_foreground(self):
        self.check_case([1]+[0]*4095, [1]*4096)

    def test_equal_classes(self):
        self.check_case([0, 1]*4, [1]*8)

    def test_all_positive(self):
        self.check_case([1]*12, [1]*12)

    def test_padding(self):
        self.check_case([1, 0, 0, 0, 0], [1, 1, 0, 0, 1])

    def test_single_lesion_and_all_negative_match_previous_v3(self):
        for target in ([1., 1., 0., 0., 0.], [0.] * 5):
            y = torch.tensor(target).view(1, 1, 1, 5)
            z = torch.linspace(-2, 2, 5).view_as(y).requires_grad_()
            m = torch.tensor([True, True, True, True, False]).view_as(y)
            w = torch.where(y.bool(), 1., float('nan'))
            _, parts = masked_loss(z, y, m, w, True)
            ref = F.binary_cross_entropy_with_logits(z[m], y[m])
            torch.testing.assert_close(parts['bce'], ref)
            torch.testing.assert_close(torch.autograd.grad(parts['bce'], z)[0],
                                       torch.autograd.grad(ref, z)[0])
        self.assertIs(train.masked_loss, masked_loss)

    def test_multilesion_uses_frozen_loader_and_preserves_class_coefficients(self):
        from ..v2.loading import Cache
        from .settings import config

        ids = np.zeros((512, 512, 1), dtype=np.uint16)
        ids[0, 0, 0] = 1
        ids[0, 2:5, 0] = 2
        raw = (ids[:, :, 0] != 0).astype(np.uint8) * 3
        valid = np.ones((512, 512), dtype=bool)
        valid[0, 7] = False
        inventory = {'synthetic': {'components': [
            {'id': 1, 'volume_mm3': 1e-12}, {'id': 2, 'volume_mm3': 1e12},
        ]}}
        with tempfile.TemporaryDirectory() as d:
            cache = Cache(d, None, config(), inventory)
            with patch.object(cache, 'arrays', return_value=(np.zeros((12,512,512), np.float32), raw)), \
                 patch.object(cache, 'components', return_value=ids), \
                 patch('experiments.unet25d.cohort.v2.loading.effective', return_value=(valid, valid, None)):
                _, y, m, w, _ = cache.sample_v2(
                    {'metadata': {'exam_id': 'synthetic'}},
                    dict(z=0, scale=512, origin=(0,0), positive=True, kind='mandatory'))
        # Clipped lesion totals 4:1, divided over 1 and 3 pixels and
        # rescaled to total foreground weight 4.
        y, m, w = [v[None, :, :1, :8] for v in (y, m, w)]
        torch.testing.assert_close(w.flatten(), torch.tensor([3.2, 1., 4/15, 4/15, 4/15, 1., 1., 1.]))
        pos, neg = y.bool(), m.bool() & ~y.bool()
        torch.testing.assert_close(w[pos].sum(), pos.sum().float())
        z = torch.tensor([1., 2., -1., -2., 0., -1., 1., 3.]).view_as(y).requires_grad_()
        _, new = masked_loss(z, y, m, w, True)
        _, old = balanced_loss(z, y, m, w, True)
        standard = F.binary_cross_entropy_with_logits(z[m.bool()], y[m.bool()])
        e = F.binary_cross_entropy_with_logits(z, y, reduction='none')
        reference = ((e[pos]*w[pos]).sum()+e[neg].sum())/7
        torch.testing.assert_close(new['bce'], reference)
        torch.testing.assert_close(new['soft_dice'], old['soft_dice'], rtol=0, atol=0)
        self.assertNotAlmostEqual(float(new['bce']), float(old['bce']))
        self.assertNotAlmostEqual(float(new['bce']), float(standard))
        new_g, old_g, standard_g = [torch.autograd.grad(b, z, retain_graph=True)[0]
                                    for b in (new['bce'], old['bce'], standard)]
        coefficient = new_g / (z.detach().sigmoid()-y)
        torch.testing.assert_close(coefficient[pos].sum(), torch.tensor(4/7))
        torch.testing.assert_close(coefficient[neg].sum(), torch.tensor(3/7))
        torch.testing.assert_close(new_g[pos], standard_g[pos]*w[pos])
        torch.testing.assert_close(new_g[neg], standard_g[neg])
        torch.testing.assert_close(old_g[pos], new_g[pos]*7/8)
        torch.testing.assert_close(old_g[neg], new_g[neg]*7/6)
        self.assertEqual(float(new_g[~m.bool()].abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, 'average one'):
            masked_loss(z,y,m,w*2)

    def test_batch_per_sample_mean(self):
        z = torch.tensor([[-2., 0., 1., 3.], [1., 2., 3., 4.]]).view(2,1,1,4).requires_grad_()
        y = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.]]).view_as(z)
        m = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool).view_as(z)
        ref = torch.stack([F.binary_cross_entropy_with_logits(a[b], c[b]) for a,b,c in zip(z,m,y)]).mean()
        _, parts = masked_loss(z, y, m, return_parts=True)
        torch.testing.assert_close(ref, parts['bce'])
        torch.testing.assert_close(torch.autograd.grad(ref, z)[0], torch.autograd.grad(parts['bce'], z)[0])

    def test_no_gt_exclusion_or_empty_validity(self):
        z = torch.zeros(1,1,1,2)
        with self.assertRaisesRegex(ValueError, 'GT excluded'):
            masked_loss(z, torch.ones_like(z), torch.zeros_like(z))
        with self.assertRaisesRegex(ValueError, 'No acquired'):
            masked_loss(z, z, z)


class IdentityTests(unittest.TestCase):
    def test_v2_identity_rejected_and_v3_round_trip(self):
        s = initial('c','i','p')
        self.assertNotEqual(science_hash(), v2_hash())
        self.assertEqual(s['data_mode'], SCHEMA)
        self.assertIn('/v3/', prefix())
        s.update(source_sha='a'*40, runtime_precision=RUNTIME_PRECISION, rank_rng=[{}, {}])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'resume.pt'
            torch.save(s,p)
            self.assertEqual(load_checkpoint(p,'c','i','p','a'*40),s)
            for change in ({'data_mode':V2_SCHEMA,'science_sha256':v2_hash()},
                           {'science_sha256':'84b8aa2a02e8b80577ac3b17b12e4a9a1f53c2894880c255bc2ec77a38a8da0a'},
                           {'source_sha':'b'*40}, {'cursor':2}):
                torch.save(dict(s, **change),p)
                with self.assertRaises(ValueError):
                    load_checkpoint(p,'c','i','p','a'*40)

    def test_numpy_partial_validation_serialization_and_resume_selection(self):
        old = [dict(exam_id='000',dice=np.float64(.3),matched_lesions=1,truth_lesions=1,empty_prediction=False,count=np.int64(2), more=[np.bool_(True)])]
        s = initial('c','i','p')
        s.update(validation_progress=old,pending_validation=True,source_sha='a'*40,
                 runtime_precision=RUNTIME_PRECISION,rank_rng=[{}, {}])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'resume.pt'
            torch.save(s,p)
            with self.assertRaisesRegex(Exception,'Weights only load failed'):
                load_checkpoint(p,'c','i','p','a'*40)
            torch.save(plain_state(s),p)
            new=load_checkpoint(p,'c','i','p','a'*40)
            self.assertEqual(new['validation_progress'],[dict(exam_id='000',dice=.3,matched_lesions=1,truth_lesions=1,empty_prediction=False,count=2,more=[True])])
            planned=[]
            def windows(shape):
                planned.append(shape)
                return [(0,0)]
            catalogue={'visits':[dict(partition='validation',metadata=dict(
                patient_id=str(i),exam_id=f'{i:03}',shape=(512,512,i+1)
            )) for i in range(45)]}
            model=torch.nn.Linear(1,1)
            # A past deadline stops before acquisition or inference. The actual
            # planner still scores every remaining exam and must skip the saved one.
            with patch('experiments.unet25d.cohort.v2.validation.windows',windows),                  patch('experiments.unet25d.cohort.v2.validation.patient_aggregates',return_value=[]):
                self.assertFalse(validate_patients(model,catalogue,
                    type('Cache',(),{'mask_counts':{}})(),None,Path(d)/'val',
                    'source','checkpoint.pt',1,0,new['validation_progress'])['complete'])
                self.assertEqual({x[2] for x in planned},set(range(2,46)))
                with self.assertRaisesRegex(ValueError,'Invalid saved'):
                    validate_patients(model,catalogue,None,None,Path(d)/'val',
                        'source','checkpoint.pt',1,0,new['validation_progress']*2)

    def test_v2_output_lease_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'private-output-lease.json'
            key = prefix()+'/live/session/0.tar'
            url = 'https://example.r2.cloudflarestorage.com/novo-nordisk-pbl-outputs/'+key
            lease = dict(expires_at=time.time()+500,slots=[dict(key=key,put=url,get=url,receipt_put=url+'.json')])
            p.write_text(json.dumps(lease))
            verify_lease(d)
            bad=copy.deepcopy(lease)
            bad['slots'][0]['put']=url.replace('/v3/','/v2/')
            p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'declared output'):
                verify_lease(d)


class EncoderTests(unittest.TestCase):
    def test_offline_encoder_uses_smp_native_channel_expansion(self):
        from safetensors.torch import save_file
        from .settings import config

        c = config()
        with tempfile.TemporaryDirectory() as d:
            snapshot = Path(d)
            base = train.smp.encoders.get_encoder('resnet34', in_channels=3, weights=None)
            (snapshot/'config.json').write_text('{}')
            save_file(base.state_dict(), snapshot/'model.safetensors')
            model = train.pretrained_model(c, snapshot)
            expected = base.conv1.weight.repeat(1, 4, 1, 1) / 4
            torch.testing.assert_close(model.encoder.conv1.weight, expected, rtol=0, atol=0)
            self.assertEqual(model.segmentation_head[0].out_channels, 1)
            self.assertTrue(all(m.momentum == c['bn_momentum']
                                for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))


class CampaignScheduleTests(unittest.TestCase):
    def test_training_completion_requires_exact_frozen_counts_and_gate(self):
        from .settings import config
        from .state import complete_phase1

        state = initial('c', 'i', 'p')
        state.update(step=146923, cursor=293846, mandatory_unique=220384,
                     replay_consumed=73462, learning_gate_passed=True)
        for change in ({'step': 146922}, {'replay_consumed': 73461},
                       {'learning_gate_passed': False}, {'pending_validation': True}):
            with self.assertRaisesRegex(ValueError, 'exact committed coverage'):
                complete_phase1(dict(state, **change), config(), 293846)
        result = complete_phase1(state, config(), 293846)
        self.assertEqual(result['status'], 'PHASE1_TRAINING_COMPLETE')
        self.assertEqual(state['scientific_stop'], result['status'])
        self.assertFalse(result['validation_performed'])
        self.assertFalse(result['phase2_launched'])
