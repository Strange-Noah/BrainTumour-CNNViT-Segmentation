import hashlib,inspect,json,os,urllib.request
from pathlib import Path
r={'gpu_allocated':False,'interactive_reused':False}
try:
 with urllib.request.urlopen('https://raw.githubusercontent.com/Kaggle/docker-python/main/patches/kaggle_secrets.py',timeout=20) as response:r['public_https_status']=response.status
except Exception as e:r['public_https_error_type']=type(e).__name__
try:
 import kaggle_web_client
 r['runtime_client_source_sha256']=hashlib.sha256(inspect.getsource(kaggle_web_client).encode()).hexdigest()
 r['runtime_secret_token_present']=bool(os.environ.get('KAGGLE_USER_SECRETS_TOKEN'))
 from kaggle_secrets import UserSecretsClient
 value=UserSecretsClient().get_secret('R2_ACCESS_KEY_ID');r['secret_available']=bool(value);del value
except Exception as e:
 r['secret_available']=False;chain=[];seen=set()
 while e is not None and id(e) not in seen:
  seen.add(id(e));row={'type':type(e).__name__}
  code=getattr(e,'code',None)
  if isinstance(code,int):row['http_status']=code
  chain.append(row);e=e.__cause__ or e.__context__
 r['sanitized_error_chain']=chain
files=list(Path('/kaggle/input').rglob('cpu-checks.json'));r['private_input_file_found']=len(files)>0
if files:
 data=json.loads(files[0].read_text());r['private_input_valid']=data.get('full_network_cpu_gradient') is True and data.get('gpu_benchmark') is False
Path('/kaggle/working/interface-probe.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
