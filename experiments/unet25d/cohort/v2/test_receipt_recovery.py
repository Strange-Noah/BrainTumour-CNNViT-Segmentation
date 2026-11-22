"""Receipt-format repair preserves end-to-end archive hashing and strict scope."""
import io,json,unittest
from types import SimpleNamespace
from .recover_preserve import read_receipt,LatestReceiptClient

class ReceiptTests(unittest.TestCase):
    def test_signed_receipt_without_metadata_sha_is_not_prepared_metadata(self):
        prefix='private/live/run';key=prefix+'/36.tar.json'
        event=dict(event='v2_private_snapshot',kind='formal_checkpoint',slot=36,key=prefix+'/36.tar',sha256='a'*64,bytes=10240)
        def client(data):return SimpleNamespace(get_object=lambda **kw:dict(Body=io.BytesIO(json.dumps(data).encode()),Metadata={}))
        self.assertEqual(read_receipt(client(event),'bucket',key,prefix),event)
        for bad in (dict(event,key='foreign/36.tar'),dict(event,sha256='invalid'),dict(event,slot=35)):
            with self.assertRaises(ValueError):read_receipt(client(bad),'bucket',key,prefix)
        with self.assertRaises(PermissionError):read_receipt(client(event),'bucket','private/active.json',prefix)
    def test_only_latest_receipt_is_replayed_not_older_checkpoint(self):
        rows=[dict(Key='private/live/run/'+str(i)+'.tar.json',LastModified=i) for i in range(37)]
        c=LatestReceiptClient(SimpleNamespace(list_objects_v2=lambda **kw:dict(Contents=rows)),'private/live/run')
        self.assertEqual(c.list_objects_v2(Prefix='private/live/run/')['Contents'],[rows[-1]])
        self.assertEqual(c.list_objects_v2(Prefix='other/')['Contents'],rows)

if __name__=='__main__':unittest.main()

class BudgetTests(unittest.TestCase):
    def test_unused_reservation_release_keeps_unsaved_compute_charged(self):
        from .transport_recovery import reconcile_reservation
        record=dict(initial_quota=dict(refresh_at='same',used_seconds=2988),reserved_seconds=43200)
        self.assertEqual(reconcile_reservation(record,dict(refresh_at='same',used_seconds=24948)),22032)
        with self.assertRaises(PermissionError):reconcile_reservation(record,dict(refresh_at='next',used_seconds=0))
        with self.assertRaises(ValueError):reconcile_reservation(record,dict(refresh_at='same',used_seconds=50000))
