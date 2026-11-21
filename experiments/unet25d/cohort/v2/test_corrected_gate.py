"""CPU release checks; no session creation or GPU work."""
import unittest,json,tempfile
from pathlib import Path
import numpy as np,torch
from .train import fixed_rows,measure
from .settings import config,science_hash
from .loading import component_exposure

class CorrectedGateTests(unittest.TestCase):
    def test_deadline_and_strict_thresholds(self):
        c=config();self.assertEqual(c['learning_gate_seconds'],600);self.assertEqual(c['learning_gate_max_steps'],4000)
        self.assertEqual(c['learning_gate_positive_probability'],.6);self.assertEqual(c['learning_gate_dice'],.1)
        self.assertEqual(c['learning_gate_negative_probability_max'],.4);self.assertEqual(c['learning_gate_negative_predicted_fraction_max'],.1)
    def test_tiny_isolation_selection_preserves_plan(self):
        inv={'e':{'components':[dict(id=1,volume_mm3=10,planes={'0':dict(voxels=22,lo=[0,0],hi=[2,11])}),dict(id=2,volume_mm3=200,planes={'0':dict(voxels=84,lo=[10,10],hi=[20,20])}),dict(id=3,volume_mm3=20,planes={'0':dict(voxels=19,lo=[200,200],hi=[205,205])})]}}
        base=dict(exam='e',z=0,positive=True,kind='replay',negative_category=None)
        plan=[dict(base,scale=512,component=2,origin=[0,0]),dict(base,scale=128,component=1,origin=[0,0]),dict(base,scale=128,component=3,origin=[150,150]),dict(base,scale=512,positive=False,negative_category='negative_enhancing'),dict(base,scale=512,positive=False,negative_category='negative_empty')]
        original=json.dumps(plan,sort_keys=True);r=fixed_rows(plan,inv,config())
        self.assertEqual(r[1]['component'],3);self.assertEqual(r[4]['component'],1);self.assertEqual(json.dumps(plan,sort_keys=True),original)
    def test_native_only_fit_cannot_pass(self):
        class Dummy(torch.nn.Module):
            def forward(self,x):return x[:,:1]
        c=config();examples=[]
        for role,positive,p in [('native_capacity',True,.95),('tiny_capacity',True,.02),('negative_enhancing',False,.01),('negative_empty',False,.01),('stress_composite',True,.01)]:
            x=torch.full((12,8,8),float(np.log(p/(1-p))));y=torch.ones(1,8,8) if positive else torch.zeros(1,8,8);mask=torch.ones_like(y);identity=dict(scale=128 if role=='tiny_capacity' else 512,gate_role=role,component=1,gate_component_masks={1:y.bool()} if positive else {})
            examples.append((x,y,mask,mask,identity))
        self.assertFalse(measure(Dummy(),examples,'cpu',c)['passed'])
