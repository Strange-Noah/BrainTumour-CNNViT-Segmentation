"""Real signed PUT/GET transport acceptance on existing private CPU runtime."""
import json,time,uuid,tempfile
from pathlib import Path
from cloud.storage import connect,verify
from .settings import prefix
from .persistence import snapshot,recover_snapshot

def run():
    c,b=connect(output=True);base=prefix()+'/live/cpu-acceptance-'+uuid.uuid4().hex;key=base+'/probe.tar'
    def signed(k,method):return c.generate_presigned_url(method,Params=dict(Bucket=b,Key=k,**({'IfNoneMatch':'*'} if method=='put_object' else {})),ExpiresIn=1800)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);inputs=root/'input';output=root/'output';inputs.mkdir();output.mkdir()
        (inputs/'private-output-lease.json').write_text(json.dumps(dict(expires_at=time.time()+1800,slots=[dict(key=key,put=signed(key,'put_object'),get=signed(key,'get_object'),receipt_put=signed(key+'.json','put_object'))])))
        (output/'transport-only.json').write_text(json.dumps(dict(cpu_only=True,no_training=True)))
        event=snapshot(output,inputs,'cpu_transport_probe')
        independent,ib=connect(output=True);verify(independent,ib,event['key'],event['sha256'])
        assert not recover_snapshot(independent,ib,event)
        print(json.dumps(dict(event='v2_cpu_transport_verified',sha256=event['sha256'],independent_readback=True,gpu_allocated=False)),flush=True)
    return True
