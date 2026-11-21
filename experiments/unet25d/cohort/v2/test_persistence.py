"""CPU regressions: signed transport corruption, immutable slots, resume monotonicity."""
import io,json,tarfile,tempfile,time,hashlib,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from types import SimpleNamespace
from .persistence import snapshot,recover_snapshot
from .settings import prefix,science_hash

class Response(io.BytesIO):status=200

class PersistenceTests(unittest.TestCase):
    def test_snapshot_signed_put_readback_and_unique_slots(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'out';inputs=root/'in';output.mkdir();inputs.mkdir();(output/'checkpoint.pt').write_bytes(b'opaque model optimizer rng')
            slots=[dict(key=prefix()+'/live/test/'+str(i),put='https://private/put'+str(i),get='https://private/get'+str(i),receipt_put='https://private/receipt'+str(i)) for i in range(2)]
            (inputs/'private-output-lease.json').write_text(json.dumps(dict(expires_at=time.time()+3600,slots=slots)))
            stored={}
            def request(req,**kw):
                if isinstance(req,str):return Response(stored[req.replace('get','put')])
                self.assertEqual(req.get_header('If-none-match'),'*');self.assertNotIn(req.full_url,stored);stored[req.full_url]=req.data;return Response()
            with patch('urllib.request.urlopen',request),redirect_stdout(io.StringIO()):
                a=snapshot(output,inputs,'formal_checkpoint');b=snapshot(output,inputs,'formal_checkpoint')
            self.assertNotEqual(a['key'],b['key']);self.assertEqual(json.loads((output/'persistence-cursor.json').read_text())['next'],2)
            self.assertEqual(a['sha256'],hashlib.sha256(stored['https://private/put0']).hexdigest())
    def test_corrupted_get_never_produces_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'private-output-lease.json').write_text(json.dumps(dict(expires_at=time.time()+3600,slots=[dict(key=prefix()+'/live/x',put='https://private/put',get='https://private/get',receipt_put='https://private/receipt')])));(p/'a.json').write_text('{}')
            with patch('urllib.request.urlopen',lambda request,**kw:Response(b'corrupt')):
                with self.assertRaises(ValueError):snapshot(p,p,'probe')
    def test_monotonic_independent_cpu_resume(self):
        class Missing(Exception):pass
        client=SimpleNamespace(exceptions=SimpleNamespace(NoSuchKey=Missing))
        def archive(step,cursor):
            data=io.BytesIO()
            with tarfile.open(fileobj=data,mode='w') as t:
                for name,body in [('training-state.json',json.dumps(dict(science_sha256=science_hash(),step=step,cursor=cursor)).encode()),('checkpoint.pt',b'opaque tensors rng optimizer')]:
                    item=tarfile.TarInfo(name);item.size=len(body);t.addfile(item,io.BytesIO(body))
            return data.getvalue()
        event=dict(key=prefix()+'/live/test/0.tar',sha256='verified',kind='formal_checkpoint')
        for step,cursor,old,success in [(1,2,None,True),(1,3,None,False),(1,2,dict(state=dict(step=2)),False)]:
            data=archive(step,cursor)
            def verify(c,b,key,sha,out):out.write(data)
            def read(*args):
                if old is None:raise Missing()
                return old
            with patch('cloud.storage.verify',verify),patch('cloud.storage.put_once',return_value={'key':'checkpoint','sha256':'checked'}),patch('experiments.unet25d.cohort.settings.read_json_record',read),patch('experiments.unet25d.cohort.v2.settings.write_json') as write:
                if success:self.assertTrue(recover_snapshot(client,'private',event));write.assert_called_once()
                else:
                    with self.assertRaises(ValueError):recover_snapshot(client,'private',event)
                    write.assert_not_called()

if __name__=='__main__':unittest.main()
