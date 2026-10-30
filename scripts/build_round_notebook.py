"""Pin the first additional round. Private input includes our own start model."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from build_policy_notebook import build


def build_round(output, archive, round_id=3):
    if round_id != 3:raise ValueError('Freeze v4/v5 only after the prior actual result has been reviewed')
    archive=Path(archive);input_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        expected={'data.npz','data-manifest.json','start-policy.json','SHA256SUMS.json'}
        if set(z.namelist())!=expected:raise ValueError('Unexpected private round input')
        checks=json.loads(z.read('SHA256SUMS.json'))
        for name in expected-{'SHA256SUMS.json'}:
            if hashlib.sha256(z.read(name)).hexdigest()!=checks[name]:raise ValueError('Private input checksum differs')
        manifest=json.loads(z.read('data-manifest.json'))
        if manifest['round_id']!=round_id:raise ValueError('Input round differs')
        if checks['start-policy.json']!=manifest['warm_start_file_sha256']:raise ValueError('Start model differs')
    build(output,input_sha)
    notebook=json.loads(output.read_text());code=''.join(notebook['cells'][1]['source'])
    def change(old,new):
        nonlocal code
        if code.count(old)!=1:raise ValueError('Base Notebook template changed: '+old)
        code=code.replace(old,new)
    change('visible*3520/3600+.05','visible*1520/3600+.05')
    start=code.index('if DATA_BASE64 is not None:')
    end=code.index('env=os.environ.copy()')
    input_code=f'''INPUT_CHECKSUMS={checks!r}
archives=list(pathlib.Path('/kaggle/input').glob('**/policy-round{round_id}-inputs.zip'))
archives+=list(pathlib.Path('/kaggle/input').glob('**/policy-round{round_id}-inputs.bundle'))
allowed={{'data.npz','data-manifest.json','start-policy.json','SHA256SUMS.json'}}
if len(archives)==1:
    data_raw=archives[0].read_bytes()
    if hashlib.sha256(data_raw).hexdigest()!=DATA_ZIP_SHA256:raise ValueError('Input ZIP checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(data_raw)) as z:
        if set(z.namelist())!=allowed or sum(v.file_size for v in z.infolist())>200_000_000:
            raise ValueError('Unexpected round input archive')
        items={{name:z.read(name) for name in allowed}}
    input_zip_verified=True
elif not archives:
    # Kaggle may expand uploaded ZIPs into individual dataset files.
    directories=[p.parent for p in pathlib.Path('/kaggle/input').glob('**/data-manifest.json')
                 if all((p.parent/name).is_file() for name in allowed)]
    if len(directories)!=1:raise RuntimeError('Attach exactly one private round input dataset')
    items={{name:(directories[0]/name).read_bytes() for name in allowed}}
    input_zip_verified=False
else:raise RuntimeError('Duplicate round input archives')
if json.loads(items['SHA256SUMS.json'])!=INPUT_CHECKSUMS:raise ValueError('Input checksum manifest differs')
data_dir=pathlib.Path('/kaggle/working/shogi-policy-v2-data');data_dir.mkdir(exist_ok=True)
for name,expected in INPUT_CHECKSUMS.items():
    if hashlib.sha256(items[name]).hexdigest()!=expected:raise ValueError('Input item checksum mismatch')
    (data_dir/name).write_bytes(items[name])
manifest=json.loads(items['data-manifest.json'])
if manifest['round_id']!={round_id}:raise ValueError('Wrong round input')
if manifest['warm_start_file_sha256']!=INPUT_CHECKSUMS['start-policy.json']:raise ValueError('Wrong starting model')
'''
    code=code[:start]+input_code+code[end:]
    change("'--device','cuda','--epochs','80','--batch-size','128','--max-wall-seconds','3300']",
           "'--device','cuda','--epochs','40','--batch-size','128','--max-wall-seconds','1200',\n"
           "         '--initial-model',str(data_dir/'start-policy.json'),'--learning-rate','0.0003',\n"
           "         '--weight-decay','0.03','--label-smoothing','0.05','--patience','6','--min-epochs','6','--seed','314161','--defer-test']")
    change('3500-(time.monotonic()-started)','1500-(time.monotonic()-started)')
    change("result={'status':training.get('status','failed')",f"result={{'round_id':{round_id},'status':training.get('status','failed')")
    change("'cpu_arena_pending':True,","'cpu_arena_pending':True,'input_zip_sha256_verified':input_zip_verified,'input_payload_sha256_verified':True,")
    code=code.replace('/kaggle/working/shogi-policy-v2',f'/kaggle/working/shogi-policy-round{round_id}')
    code=code.replace('policy-v2-return.txt',f'policy-round{round_id}-return.txt')
    code=code.replace('Attach exactly one private policy-v2-gpu-inputs.zip CPU output',
                      f'Attach exactly one private policy-round{round_id}-inputs.zip CPU output')
    compile(code,str(output),'exec')
    notebook['cells'][1]['source']=code.splitlines(keepends=True)
    notebook['metadata']['round_id']=round_id
    notebook['metadata']['input_contains_own_start_model']=True
    notebook['metadata']['input_embedding']=False
    notebook['metadata']['start_weights_sha256']=manifest['warm_start_weights_sha256']
    notebook['cells'][0]['source']=[
        '# 電竜戦 追加1回目/3（v3）：過学習対策\n',
        f'非公開・Internet OFF。非公開Inputへ policy-round{round_id}-inputs.zip を追加してください。\n',
        '自分のv2重みから追加学習します。モデルや入力データをこの公開Notebookへ埋め込んでいません。\n',
        '無料GPU残量をUIで確認し、REMAINING_FREE_GPU_HOURSへ入力。既存PyTorch/CUDAを更新しません。\n',
        '最大40epoch・validation早期終了。testは計算せず、全3回の方法確定後にCPUで最終検証します。\n',
        '学習累計1200秒・全体1500秒＋終了猶予15秒。保存後すぐセッションOFF。対局はCPUで別実行。\n']
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps({'round_id':round_id,'notebook':str(output),'notebook_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'source_commit':notebook['metadata']['policy_source']['commit'],
        'source_zip_sha256':notebook['metadata']['source_zip_sha256'],'input_zip_sha256':input_sha,
        'input_bytes':archive.stat().st_size,'data_counts':manifest['counts']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-archive',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('notebooks/kaggle_policy_round3.ipynb'))
    a=p.parse_args();build_round(a.output,a.input_archive)
