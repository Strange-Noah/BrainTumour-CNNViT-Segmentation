"""CPU-only authorization checks; no provider SDK, credentials or GPU calls."""
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from cloud.resolve_workstream import resolve
from cloud.workstream_route import route

SHA=json.loads(Path('cloud/workstreams.json').read_text())['unet25d']['source_sha']

class RoutingTests(unittest.TestCase):
    def call(self, **kw):
        args=dict(name='unet25d',source='',actor='Strange-Noah',ref='refs/heads/main',event='workflow_dispatch',mode='collect')
        args.update(kw)
        return route(**args)

    def test_scheduled_disabled_without_personal_code(self):
        with patch('cloud.workstream_route.resolve', side_effect=AssertionError('must not resolve model')):
            self.assertEqual(self.call(event='schedule', actor='scheduler')['enabled'],'false')

    def test_manual_recovery_disabled(self):
        self.assertEqual(self.call(mode='recover')['enabled'],'false')

    def test_formal_refuses_old_authorization(self):
        with self.assertRaises(PermissionError): self.call(mode='formal')

    def test_other_actor_refused(self):
        with self.assertRaises(PermissionError): self.call(actor='someone-else')

    def test_other_dispatch_ref_refused(self):
        with self.assertRaises(PermissionError): self.call(ref='refs/heads/model/unet25d-kaggle')

    def test_missing_resolver_source_refused(self):
        with self.assertRaises(PermissionError): resolve('unet25d','','Strange-Noah')

    def test_unknown_source_refused(self):
        with self.assertRaises(PermissionError): self.call(source='1'*40)

    def test_registry_ancestor_required(self):
        with patch('cloud.resolve_workstream.subprocess.run', side_effect=subprocess.CalledProcessError(1,'git')):
            with self.assertRaises(subprocess.CalledProcessError): self.call()

    def test_registered_manual_collect(self):
        with patch('cloud.resolve_workstream.subprocess.run') as check:
            r=self.call()
        self.assertEqual(r['source_sha'],SHA)
        self.assertEqual(r['approved_ref'],'origin/model/unet25d-kaggle')
        check.assert_called_once_with(['git','merge-base','--is-ancestor',SHA,'origin/model/unet25d-kaggle'],check=True)

    def test_arbitrary_resources_refused(self):
        with patch('cloud.resolve_workstream.subprocess.run'):
            for key in ('dataset','kernel'):
                with self.assertRaises(PermissionError): self.call(**{key:'other/resource'})

    def test_cpu_shared_ancestor(self):
        with patch('cloud.workstream_route.subprocess.run',return_value=subprocess.CompletedProcess([],0)):
            self.assertEqual(self.call(cpu=True,source='2'*40)['approved_ref'],'origin/main')

    def test_cpu_registered_alternative(self):
        def git(args,check=False):
            return subprocess.CompletedProcess(args,1 if args[-1]=='origin/main' else 0)
        with patch('subprocess.run',side_effect=git):
            self.assertEqual(self.call(cpu=True,source=SHA)['source_sha'],SHA)
            with self.assertRaises(PermissionError):self.call(cpu=True,source='1'*40)

    def test_cpu_invalid_sha_refused(self):
        with self.assertRaises(PermissionError):self.call(cpu=True,source='main')

    def test_workflow_order_and_secret_boundary(self):
        import yaml
        w=yaml.safe_load(Path('.github/workflows/kaggle-benchmark.yml').read_text())
        steps=w['jobs']['orchestrate']['steps']
        route_index=next(i for i,s in enumerate(steps) if s.get('id')=='route')
        checkout_index=next(i for i,s in enumerate(steps) if s.get('name')=='Check out exact approved workstream')
        self.assertLess(route_index,checkout_index)
        for step in steps[:-1]: self.assertNotIn('secrets.',json.dumps(step))
        self.assertIn("steps.route.outputs.enabled == 'true'",steps[-1]['if'])
        self.assertIn('steps.route.outputs.source_sha',steps[-1]['env']['NOVO_SOURCE_SHA'])
        self.assertEqual(w['permissions'],{'contents':'read'})
        cpu=yaml.safe_load(Path('.github/workflows/project-execute.yml').read_text())['jobs']['execute']['steps']
        self.assertEqual(cpu[0]['with']['ref'],'main')
        auth=next(i for i,s in enumerate(cpu) if s.get('id')=='route')
        approved=next(i for i,s in enumerate(cpu) if s.get('name')=='Check out authorized immutable source')
        self.assertLess(auth,approved)
        for step in cpu[:-1]:self.assertNotIn('secrets.',json.dumps(step))

if __name__=='__main__':unittest.main()
