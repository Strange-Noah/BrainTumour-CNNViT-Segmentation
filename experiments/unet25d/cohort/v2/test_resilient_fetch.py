import hashlib,io,tempfile,time,unittest,urllib.error
from pathlib import Path
from unittest.mock import patch
from .resilient_fetch import SignedFetch
class FetchTests(unittest.TestCase):
    def run_fetch(self,effects,expected=None,expired=False):
        data=b'verified prepared bytes';sha=hashlib.sha256(data).hexdigest();expected=expected or sha
        lease=dict(expires_at=time.time()+(-1 if expired else 300),objects={'key':dict(url='https://account.r2.cloudflarestorage.com/private?signature=never-log',sha256=expected)})
        with tempfile.TemporaryDirectory() as d,patch('urllib.request.urlopen',side_effect=effects) as opened,patch('time.sleep'):
            p=Path(d)/'object';fetch=SignedFetch(lease)
            try:fetch('key',expected,p)
            except Exception:
                self.assertFalse(p.exists());self.assertFalse(list(Path(d).glob('*.partial-*')));raise
            self.assertEqual(p.read_bytes(),data);return opened.call_count
    def test_transient_service_then_success(self):
        e=urllib.error.HTTPError('private',503,'service',{},None)
        self.assertEqual(self.run_fetch([e,io.BytesIO(b'verified prepared bytes')]),2)
    def test_timeout_then_success(self):
        self.assertEqual(self.run_fetch([TimeoutError(),io.BytesIO(b'verified prepared bytes')]),2)
    def test_integrity_auth_expiry_and_retry_cap_are_fatal(self):
        for effects,kw in (([io.BytesIO(b'corrupt')],{}),([urllib.error.HTTPError('private',403,'auth',{},None)],{}),([] ,{'expired':True}),([TimeoutError()]*3,{})):
            with self.subTest(kw=kw),self.assertRaises(RuntimeError):self.run_fetch(effects,**kw)
if __name__=='__main__':unittest.main()
