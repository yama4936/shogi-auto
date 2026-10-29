"""CPU-only reevaluation of immutable models from the completed Kaggle run.

Requires private files from that run: initial-model.json, best-model.json,
data.npz, data-manifest.json, config.json (training-report.json optional).
No training, downloads, API calls or GPU activation are performed.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from shogi_auto.learning.arena import evaluate_models
from shogi_auto.learning.common import Budget, sha256, source_identity, write_json
from shogi_auto.learning.features import FEATURES, LinearEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--move-seconds', type=float, default=0.05)
    parser.add_argument('--pairs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--max-wall-seconds', type=float, default=1800)
    args = parser.parse_args()
    if not 0.005 <= args.move_seconds <= .5 or not 30 <= args.pairs <= 200:
        raise ValueError('Reevaluation requires at least 30 opening pairs and a bounded equal move budget')
    source = args.source_dir.resolve()
    target = args.run_dir.resolve()
    if source == target:
        raise ValueError('Use a new run directory; do not modify the original experiment')
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'data-manifest.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    config.update(arena_seed=args.seed, arena_pairs=args.pairs, move_seconds=args.move_seconds, workers=2)
    identity = {'stage': 'reevaluation_only_no_training', 'source': source_identity(),
                'training_data_sha256': manifest['corpus_sha256'], 'source_data_npz_sha256': sha256(source / 'data.npz'),
                'config': config, 'source_models': {name: sha256(source / name)
                                                  for name in ('initial-model.json', 'best-model.json')},
                'selection_policy': 'immutable GPU-selected model; no selection using previous or new arena outcomes'}
    for name in ('initial-model.json', 'best-model.json'):
        LinearEvaluator.load(source / name)
        model = json.loads((source / name).read_text())
        if model['metadata']['data_sha256'] != manifest['corpus_sha256']:
            raise ValueError('Model and corpus provenance mismatch')
    with np.load(source / 'data.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('x', 'y', 'split', 'game_id')}
    if (arrays['x'].ndim != 2 or arrays['x'].shape[1] != FEATURES or
            not np.isfinite(arrays['x']).all() or not np.isfinite(arrays['y']).all() or
            set(arrays['split'].tolist()) != {'train', 'validation', 'test'}):
        raise ValueError('Unexpected or invalid source data')
    path = target / 'reevaluation-settings.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Reevaluation settings changed; use another new directory')
    write_json(path, identity)
    for name in ('initial-model.json', 'best-model.json'):
        destination = target / name
        if destination.exists() and sha256(destination) != identity['source_models'][name]:
            raise ValueError('Existing model differs; refusing to overwrite it')
        shutil.copyfile(source / name, destination)
    write_json(target / 'config.json', config)
    budget = Budget(target, args.max_wall_seconds, visible_gpus=0)
    try:
        report = evaluate_models(target, arrays, config, budget)
        print(json.dumps({'status': 'evaluated', 'comparisons': report['comparisons'],
                          'improvement_evidence': report['improvement_evidence']}, ensure_ascii=False), flush=True)
    finally:
        budget.save()


if __name__ == '__main__':
    main()
