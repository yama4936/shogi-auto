"""Assemble a new supervised refinement without replaying prior held-out rows."""
import hashlib
from pathlib import Path
import shutil

from .common import sha256, source_identity, write_json
from .network import load_network, weights_digest
from .train import read_data


def canonical_key(x):
    return min(hashlib.sha256(x[:42].tobytes()).hexdigest(),
               hashlib.sha256(x[:42, :, ::-1].copy().tobytes()).hexdigest())


def assemble(fresh_dir, replay_dir, initial_model, destination, *, exclude_data=()):
    """Replay TRAIN only; preserve fresh validation/test and every group boundary.

    Exclusion arrays may contain diagnostic or future evaluation states. They
    are never used as labels. The full production dataset validator runs last.
    """
    import numpy as np
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Use a new immutable refinement directory')
    origin = source_identity()
    if origin['dirty']:
        raise ValueError('Pin refinement source before assembling training input')
    fresh, manifest = read_data(fresh_dir)
    replay, previous = read_data(replay_dir)
    if manifest.get('trajectory_model_sha256') != sha256(initial_model):
        raise ValueError('Refinement must use the same frozen trajectory and initial model')
    forbidden, exclusions = set(), {}
    for raw_path in exclude_data:
        path = Path(raw_path)
        exclusions[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            if archive['x'].ndim != 4 or archive['x'].shape[1:] != (43, 9, 9):
                raise ValueError('Unsupported refinement exclusion features')
            forbidden.update(canonical_key(x) for x in archive['x'])
    if any(canonical_key(x) in forbidden for x in fresh['x']):
        raise ValueError('Fresh data overlaps diagnostics or fixed evaluation states')
    train_ids = np.flatnonzero(replay['split'] == 'train')
    keep = np.array([i for i in train_ids if canonical_key(replay['x'][i]) not in forbidden], dtype=np.int64)
    arrays = {key: np.concatenate([fresh[key], replay[key][keep]]) for key in fresh}
    # Assign replay groups above the largest fresh group, retaining grouping.
    replay_groups = {value: index + int(fresh['game_id'].max()) + 1
                     for index, value in enumerate(sorted(set(replay['game_id'][keep].tolist())))}
    arrays['game_id'][len(fresh['game_id']):] = [replay_groups[value] for value in replay['game_id'][keep]]
    destination.mkdir(parents=True)
    np.savez_compressed(destination/'data.npz', **arrays)
    result = {**manifest, 'source': origin, 'generation_source': manifest['source'],
              'experiment': 'supervised-tactical-refinement-v1',
              'npz_sha256': sha256(destination/'data.npz'), 'rows': len(arrays['x']),
              'counts': {split: int((arrays['split'] == split).sum()) for split in ('train', 'validation', 'test')},
              'fresh_npz_sha256': manifest['npz_sha256'], 'replay_npz_sha256': previous['npz_sha256'],
              'replayed_train_rows': len(keep), 'replayed_validation_rows': 0, 'replayed_test_rows': 0,
              'excluded_replay_train_rows': len(train_ids)-len(keep), 'excluded_state_arrays': exclusions,
              'initial_model_sha256': sha256(initial_model),
              'initial_weights_sha256': weights_digest(load_network(initial_model)),
              'selection': 'fresh validation only; no arena or test selection',
              'split_policy': 'fresh whole games plus remapped prior TRAIN groups; canonical/mirror uniqueness required'}
    write_json(destination/'data-manifest.json', result)
    read_data(destination)
    shutil.copyfile(initial_model, destination/'start-policy.json')
    write_json(destination/'SHA256SUMS.json', {name: sha256(destination/name)
                                             for name in ('data.npz', 'data-manifest.json', 'start-policy.json')})
    return result
