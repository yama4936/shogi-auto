"""Build a source-pinned, offline, training-only Kaggle Notebook.

Run after committing the policy implementation and producing the private CPU
data archive. Neither third-party weights nor generated data enter this file.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import zipfile

from shogi_auto.policy.common import source_files, source_identity

ROOT=Path(__file__).resolve().parents[1]


def build(output, data_sha):
    if len(data_sha)!=64 or any(c not in '0123456789abcdef' for c in data_sha):
        raise ValueError('Supply the actual private CPU data archive SHA256')
    origin=source_identity()
    if origin['dirty']:raise ValueError('Commit the code first; production bundles require a clean tree')
    origin={**origin,'offline_bundle':True}
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as z:
        for path in source_files(ROOT):z.write(path,str(path.relative_to(ROOT)))
        distribution=importlib.metadata.distribution('python-shogi')
        if distribution.version!='1.1.1':raise ValueError('Reviewed python-shogi version is required')
        for file in distribution.files:
            if str(file).startswith('shogi/') and str(file).endswith('.py'):
                z.write(distribution.locate_file(file),'vendor/'+str(file))
        z.write(ROOT/'docs/licenses/python-shogi-GPL-3.0.txt','licenses/python-shogi-GPL-3.0.txt')
        z.writestr('SOURCE_POLICY.json',json.dumps(origin,ensure_ascii=False,indent=2)+'\n')
    raw=memory.getvalue();encoded=base64.b64encode(raw).decode();bundle_sha=hashlib.sha256(raw).hexdigest()
    code=r'''import base64, hashlib, json, os, pathlib, subprocess, sys, time, zipfile, io
started=time.monotonic()
# Fill this only after checking the current Kaggle free quota in the UI.
REMAINING_FREE_GPU_HOURS=None
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
if REMAINING_FREE_GPU_HOURS is None:raise RuntimeError('Check free GPU quota before running')
if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no CPU fallback')
visible=torch.cuda.device_count()
if visible not in (1,2):raise RuntimeError('Reviewed first experiment allows at most two allocated GPUs')
if REMAINING_FREE_GPU_HOURS < visible*3520/3600+.05:raise RuntimeError('Insufficient confirmed free GPU allowance')
if str(torch.__version__).split('+')[0]!='2.11.0':raise RuntimeError('PyTorch differs from the reviewed environment; do not auto-upgrade')
raw=base64.b64decode(SOURCE_BASE64,validate=True)
assert hashlib.sha256(raw).hexdigest()==SOURCE_ZIP_SHA256
code_dir=pathlib.Path('/kaggle/working/shogi-policy-v2-code')
code_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for entry in z.infolist():
        p=pathlib.PurePosixPath(entry.filename)
        if p.is_absolute() or '..' in p.parts or entry.file_size>2_000_000:raise ValueError('Invalid source bundle path/size')
    z.extractall(code_dir)
origin=json.loads((code_dir/'SOURCE_POLICY.json').read_text())
assert origin['commit']==PINNED_CODE_COMMIT
archives=list(pathlib.Path('/kaggle/input').glob('**/policy-v2-gpu-inputs.zip'))
if len(archives)!=1:raise RuntimeError('Attach exactly one private policy-v2-gpu-inputs.zip CPU output')
archive=archives[0]
if hashlib.sha256(archive.read_bytes()).hexdigest()!=DATA_ZIP_SHA256:raise ValueError('Private CPU data archive checksum mismatch')
data_dir=pathlib.Path('/kaggle/working/shogi-policy-v2-data');data_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    allowed={'data.npz','data-manifest.json','SHA256SUMS.json'}
    if set(z.namelist())!=allowed or sum(e.file_size for e in z.infolist())>200_000_000:
        raise ValueError('Unexpected private dataset archive')
    checks=json.loads(z.read('SHA256SUMS.json'))
    for name in ('data.npz','data-manifest.json'):
        if hashlib.sha256(z.read(name)).hexdigest()!=checks[name]:raise ValueError('Dataset item checksum mismatch')
        (data_dir/name).write_bytes(z.read(name))
env=os.environ.copy();env['PYTHONPATH']=str(code_dir/'src')+os.pathsep+str(code_dir/'vendor')
env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
destination=pathlib.Path('/kaggle/working/shogi-policy-v2');destination.mkdir(exist_ok=True)
command=[sys.executable,'-m','shogi_auto.policy.train','--data-dir',str(data_dir),'--run-dir',str(destination),
         '--device','cuda','--epochs','80','--batch-size','128','--max-wall-seconds','3300']
log=destination/'training-process.log';timed_out=False
with log.open('a') as stream:
    process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    try:process.wait(timeout=max(1,3500-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
        timed_out=True;process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
report_path=destination/'training-report.json'
training=json.loads(report_path.read_text()) if report_path.exists() else {}
result={'status':training.get('status','failed'),'exit_code':process.returncode,'watchdog_triggered':timed_out,
        'wall_seconds':time.monotonic()-started,'source':origin,'source_zip_sha256':SOURCE_ZIP_SHA256,
        'data_zip_sha256':DATA_ZIP_SHA256,'visible_gpus':visible,'gpu_names':[torch.cuda.get_device_name(i) for i in range(visible)],
        'gpu_training_verified':bool(training.get('gpu_training_verified',False)),
        'cuda_optimizer_steps_total':training.get('cuda_optimizer_steps_total',0),
        'strength_improvement_verified':False,'cpu_arena_pending':True,
        'note':'Save private outputs and switch the session OFF immediately; this Notebook does no CPU arena or teacher generation.'}
if process.returncode!=0:result['status']='failed'
(destination/'run-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
print(log.read_text()[-5000:])
'''
    constants=(f'SOURCE_BASE64={encoded!r}\nSOURCE_ZIP_SHA256={bundle_sha!r}\n'
               f'PINNED_CODE_COMMIT={origin["commit"]!r}\nDATA_ZIP_SHA256={data_sha!r}\n')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                 'policy_source':origin,'source_zip_sha256':bundle_sha,'private_data_zip_sha256':data_sha},
              'cells':[{'cell_type':'markdown','metadata':{},'source':[
                  '# shogi-auto 方策・価値モデル v2（学習のみ）\n',
                  '非公開・Internet OFF。CPU生成済みの非公開データZIPをInputへ追加してください。\n',
                  '無料GPU残量をUIで確認して下の値を入力。上限は学習3300秒、全体3500秒＋終了猶予15秒。\n',
                  '終了後はOutputを非公開保存し、ただちにセッションOFF。対局評価はCPUで別に行います。\n',
                  '損失低下・CUDA更新だけを棋力向上としません。既存のPyTorch/CUDAは更新しません。\n']},
                       {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                        'source':(constants+code).splitlines(keepends=True)}]}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps({'notebook':str(output),'source_commit':origin['commit'],'source_zip_sha256':bundle_sha,
                      'source_sha256':origin['source_sha256'],'private_data_zip_sha256':data_sha,'bundle_bytes':len(raw)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'notebooks/kaggle_policy_v2.ipynb')
    p.add_argument('--data-archive-sha256',required=True)
    a=p.parse_args();build(a.output,a.data_archive_sha256)
