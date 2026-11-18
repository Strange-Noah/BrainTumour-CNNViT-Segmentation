"""Run from an authenticated private Kaggle session; never exports credentials."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','prepare','campaign']);parser.add_argument('--cpu-artifacts',type=Path);args=parser.parse_args()
    artifact_mode=os.environ.get('NOVO_ARTIFACT_MODE')=='kaggle'
    if not artifact_mode:
        from kaggle_secrets import UserSecretsClient
        client=UserSecretsClient();availability={}
        for name in ['R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY','R2_OUTPUT_ACCESS_KEY_ID','R2_OUTPUT_SECRET_ACCESS_KEY']:
            if not os.environ.get(name):os.environ[name]=client.get_secret(name)
            availability[name]=bool(os.environ[name])
        if not all(availability.values()):raise RuntimeError('Missing scoped credentials')
        os.environ.update(R2_BUCKET='novo-nordisk-pbl',R2_OUTPUT_BUCKET='novo-nordisk-pbl-outputs')
        if not os.environ.get('R2_ENDPOINT_URL'):raise ValueError('Set R2_ENDPOINT_URL')
    if args.mode=='preflight':return
    work=Path('/kaggle/working/novo-unet25d');work.mkdir(exist_ok=True);work.chmod(0o700)
    if args.mode=='prepare':os.environ['HF_HOME']=str(work/'hf')
    else:
        if not args.cpu_artifacts:raise ValueError('Mount completed private CPU outputs before starting GPU session')
        for name in ['config.json','cache-receipts.json','cache_integrity.stage.json']:
            shutil.copy2(args.cpu_artifacts/name,work/name)
        (work/'prepared').symlink_to(args.cpu_artifacts/'prepared',target_is_directory=True)
        os.environ['HF_HOME']=str(args.cpu_artifacts/'hf');os.environ['HF_HUB_OFFLINE']='1'
    os.environ['NOVO_SOURCE_SHA']=os.environ.get('NOVO_SOURCE_SHA') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    launch=dict(source_sha=os.environ['NOVO_SOURCE_SHA'],started=time.time(),mode=args.mode)
    os.environ['NOVO_RUN_ID']=hashlib.sha256(json.dumps(launch,sort_keys=True).encode()).hexdigest()[:16]
    (work/'launch.json').write_text(json.dumps(dict(**launch,run_id=os.environ['NOVO_RUN_ID'])))
    process=subprocess.Popen([sys.executable,'-m','experiments.unet25d.pipeline',args.mode,str(work)])
    try:code=process.wait(timeout=min(3500,int(os.environ.get('NOVO_GPU_BUDGET_SECONDS','3600'))-60) if args.mode=='campaign' else 7200)
    except subprocess.TimeoutExpired:
        import psutil
        for child in reversed(psutil.Process(process.pid).children(recursive=True)):
            try:child.kill()
            except psutil.NoSuchProcess:pass
        process.kill();process.wait();raise
    if code:raise RuntimeError('Pipeline failed; inspect private stage completion records')
if __name__=='__main__':main()
