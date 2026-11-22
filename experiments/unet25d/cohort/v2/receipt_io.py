"""Signed receipt schema; archive bytes still require independent SHA verification."""
import json,re

def read_receipt(client,bucket,key,live_prefix):
    if not key.startswith(live_prefix+'/') or not key.endswith('.tar.json'):raise PermissionError('Unexpected GPU receipt scope')
    response=client.get_object(Bucket=bucket,Key=key)
    try:raw=response['Body'].read(32769)
    finally:response['Body'].close()
    if len(raw)>32768:raise ValueError('GPU receipt size')
    event=json.loads(raw)
    if event.get('event')!='v2_private_snapshot' or event.get('kind') not in ('startup_transport_probe','gate1','formal_checkpoint'):raise ValueError('GPU receipt schema')
    if not isinstance(event.get('slot'),int) or event['slot']<0:raise ValueError('GPU receipt slot')
    if not isinstance(event.get('key'),str) or event['key']+'.json'!=key or event['key']!=live_prefix+'/'+str(event['slot'])+'.tar':raise ValueError('GPU receipt object binding')
    if not re.fullmatch('[0-9a-f]{64}',event.get('sha256','')) or not isinstance(event.get('bytes'),int) or not 0<event['bytes']<2*1024**3:raise ValueError('GPU archive checksum/size schema')
    # A receipt is uploaded via a signed PUT, without metadata.sha256. It is
    # not prepared metadata. recover_snapshot MUST independently hash the full
    # archive before any checkpoint/pointer write; global metadata checks stay.
    return event

class LatestReceiptClient:
    def __init__(self,client,live_prefix):self.client=client;self.live_prefix=live_prefix
    def __getattr__(self,name):return getattr(self.client,name)
    def list_objects_v2(self,**kwargs):
        response=self.client.list_objects_v2(**kwargs)
        if kwargs.get('Prefix')==self.live_prefix+'/':
            rows=[r for r in response.get('Contents',[]) if r['Key'].endswith('.tar.json')]
            response=dict(response,Contents=[max(rows,key=lambda r:r['LastModified'])] if rows else [])
        return response

