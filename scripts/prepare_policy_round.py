"""Assemble a private round input; only the prior TRAIN split can be replayed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from shogi_auto.policy.common import sha256,source_identity,write_json
from shogi_auto.policy.network import load_network,weights_digest
from shogi_auto.policy.train import read_data
from shogi_auto.policy.tournament import OPENING,RULES_URL,RULES_SHA256


def prepare(fresh_dir, replay_dir, model_path, destination, round_id, previous_review=None):
    if round_id not in (3,4,5):raise ValueError('Exactly the three authorized additional rounds are supported')
    destination=Path(destination)
    if destination.exists():raise ValueError('Use a new immutable round input directory')
    review_sha = None
    if round_id > 3:
        if previous_review is None:raise ValueError('Review the completed preceding learning/evaluation round first')
        review=json.loads(Path(previous_review).read_text())
        if (review['round_id']!=round_id-1 or not review['evaluation_complete'] or
                review['training']['cuda_optimizer_steps_total']<=0 or not review['test_evaluation_deferred']):
            raise ValueError('Previous round is incomplete or its test was not deferred')
        if review['selected_weights_sha256']!=weights_digest(load_network(model_path)):
            raise ValueError('Warm start differs from the validation-selected preceding model')
        review_sha=sha256(previous_review)
    fresh,manifest=read_data(fresh_dir);replay,prior=read_data(replay_dir)
    if not manifest['config'].get('seed_namespace') or manifest['config'].get('opening_prefix')!=list(OPENING):
        raise ValueError('Fresh round data needs an independent seed namespace and the required opening')
    keep=replay['split']=='train'
    arrays={k:np.concatenate([fresh[k],replay[k][keep]]) for k in fresh}
    # IDs are provenance groups, not array offsets; keep the prior training games
    # in a separate namespace from fresh train/validation/test games.
    arrays['game_id'][len(fresh['game_id']):]+=1_000_000*round_id
    destination.mkdir(parents=True)
    np.savez_compressed(destination/'data.npz',**arrays)
    counts={s:int((arrays['split']==s).sum()) for s in ('train','validation','test')}
    origin=source_identity()
    if origin['dirty']:raise ValueError('Pin the preparation code before assembling production input')
    model=load_network(model_path)
    result={**manifest,'source':origin,'generation_source':manifest['source'],
            'round_id':round_id,'created_at':datetime.now(timezone.utc).isoformat(),
            'rows':len(arrays['x']),'counts':counts,'npz_sha256':sha256(destination/'data.npz'),
            'fresh_input_sha256':manifest['npz_sha256'],'replayed_training_sha256':prior['npz_sha256'],
            'replayed_training_rows':int(keep.sum()),'replayed_validation_rows':0,'replayed_test_rows':0,
            'warm_start_file_sha256':sha256(model_path),'warm_start_weights_sha256':weights_digest(model),
            'previous_round_review_sha256':review_sha,
            'rules':{'url':RULES_URL,'sha256':RULES_SHA256,'teacher_use':'before tournament only; no external parameter transfer'},
            'test_evaluation':'deferred; not used for this round checkpoint selection or next-round planning',
            'split_policy':'fresh game groups; prior TRAIN rows only; globally deduplicated canonical states required'}
    result['fresh_corpus_sha256']=result.pop('corpus_sha256')
    result['fresh_sampling_counts']=result.pop('sampling_counts')
    result['fresh_games']=result.pop('games')
    result['distinct_game_groups']=len(set(arrays['game_id'].tolist()))
    write_json(destination/'data-manifest.json',result)
    # Re-use the production validator to reject split leakage and mirrored rows.
    read_data(destination)
    shutil.copyfile(model_path,destination/'start-policy.json')
    names=['data.npz','data-manifest.json','start-policy.json']
    checks={name:sha256(destination/name) for name in names}
    write_json(destination/'SHA256SUMS.json',checks)
    archive=destination/f'policy-round{round_id}-inputs.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names+['SHA256SUMS.json']:z.write(destination/name,name)
    return {'archive':str(archive.resolve()),'bytes':archive.stat().st_size,'sha256':sha256(archive),
            'counts':counts,'fresh_rows':len(fresh['x']),'replayed_train_rows':int(keep.sum()),
            'warm_start_weights_sha256':result['warm_start_weights_sha256'],'source':origin}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh-dir',type=Path,required=True);p.add_argument('--replay-dir',type=Path,required=True)
    p.add_argument('--initial-model',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--round-id',type=int,required=True)
    p.add_argument('--previous-review',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.fresh_dir,a.replay_dir,a.initial_model,a.destination,a.round_id,a.previous_review),indent=2))
