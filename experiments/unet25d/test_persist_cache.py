"""Offline tests: interrupted-chunk reuse and independent corruption detection."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from botocore.exceptions import ClientError
from . import persist_cache as p
class Body(io.BytesIO):
    def iter_chunks(self,chunk_size):
        while block:=self.read(chunk_size):yield block
class Store:
    def __init__(self):self.objects={};self.puts=0
    def head_object(self,Bucket,Key):
        if Key not in self.objects:raise ClientError({'ResponseMetadata':{'HTTPStatusCode':404},'Error':{'Code':'404'}},'HeadObject')
        data=self.objects[Key];return dict(ContentLength=len(data),Metadata={'sha256':p.sha(data)})
    def put_object(self,Bucket,Key,Body,**kwargs):self.objects[Key]=Body;self.puts+=1
    def get_object(self,Bucket,Key):return dict(Body=Body(self.objects[Key]))
class Transfer(unittest.TestCase):
    def test_resume_and_remote_corruption(self):
        store=Store()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'config.json').write_text('{}');(root/'data').write_bytes(b'abcdefghijk')
            files={'config.json':root/'config.json','prepared/data.npy':root/'data'}
            with patch.object(p,'inventory',return_value=(files,'a'*40)),patch.object(p,'connect',return_value=store),patch.object(p,'CHUNK',4),patch.object(p.subprocess,'run'):
                p.upload(root);writes=store.puts;p.upload(root);self.assertEqual(writes,store.puts)
            receipt=json.loads((root/'r2-cache-upload.json').read_text())
            with patch.object(p,'connect',return_value=store),contextlib.redirect_stdout(io.StringIO()):
                p.verify(receipt['prefix'],receipt['manifest_sha256'])
                key=next(k for k in store.objects if '/chunks/' in k);store.objects[key]=b'corrupt'
                with self.assertRaises(ValueError):p.verify(receipt['prefix'],receipt['manifest_sha256'])
    def test_existing_conflict_not_overwritten(self):
        store=Store();store.objects['key']=b'old'
        with self.assertRaises(ValueError):p.put(store,'key',b'new')
        self.assertEqual(store.objects['key'],b'old')
if __name__=='__main__':unittest.main()
