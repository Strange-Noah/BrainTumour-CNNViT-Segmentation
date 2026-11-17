"""Paste/run in authenticated Kaggle CPU session. Pins code; never requests GPUs."""
import io,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
import torch
if torch.cuda.is_available():raise RuntimeError('Run CPU preparation before requesting GPU allocation')
from kaggle_secrets import UserSecretsClient
secrets=UserSecretsClient()
for name in ['R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY','R2_OUTPUT_ACCESS_KEY_ID','R2_OUTPUT_SECRET_ACCESS_KEY']:
    try:os.environ[name]=secrets.get_secret(name)
    except Exception as error:raise RuntimeError(f'Secret unavailable: {name}; {type(error).__name__}') from None
sha='4e9c94657e28f2b369523181fd12d9cb26dc3b83'
root=Path('/kaggle/working')/('novo-unet25d-source-'+sha[:8]);root.mkdir(exist_ok=True)
url=f'https://github.com/Strange-Noah/BrainTumour-CNNViT-Segmentation/archive/{sha}.zip'
with urllib.request.urlopen(url,timeout=60) as response:archive=zipfile.ZipFile(io.BytesIO(response.read()))
for item in archive.infolist():
    relative=Path(*Path(item.filename).parts[1:]);destination=root/relative
    if not destination.resolve().is_relative_to(root.resolve()):raise RuntimeError('Invalid source archive path')
    if item.is_dir():destination.mkdir(parents=True,exist_ok=True)
    else:
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(archive.read(item))
os.environ['NOVO_SOURCE_SHA']=sha
os.environ.setdefault('R2_ENDPOINT_URL','https://fec4311cb2caaffaf5e1742997da99b3.r2.cloudflarestorage.com')
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'experiments/unet25d/requirements.txt')],check=True)
subprocess.run([sys.executable,'-m','experiments.unet25d.kaggle_launcher','prepare'],cwd=root,check=True)
print('CPU cache preparation complete; no GPU was allocated by this entry point.')
