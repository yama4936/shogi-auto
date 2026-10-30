"""Replay a completed development arena and verify validation parity, without test metrics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from shogi_auto.game import Game
from shogi_auto.policy.common import sha256, source_identity, write_json
from shogi_auto.policy.network import load_network, weights_digest
from shogi_auto.policy.train import read_data


def audit(directory, data_directory, commitment):
    directory = Path(directory)
    training = json.loads((directory / 'training-report.json').read_text())
    arena = json.loads((directory / 'policy-arena-report.json').read_text())
    opening_record = json.loads(Path(commitment).read_text())
    assert arena['complete'] and arena['status'] == 'evaluated'
    assert not arena['functional_only'] and training['test_evaluation_deferred']
    assert all('test' not in training[key] for key in ('initial_metrics', 'selected_metrics'))
    origin = source_identity()
    assert not origin['dirty']
    assert origin['source_sha256'] == arena['source']['source_sha256']
    assert arena['source']['commit'] == opening_record['frozen_source']['commit']
    assert arena['source']['source_sha256'] == opening_record['frozen_source']['source_sha256']
    assert arena['openings'] == opening_record['openings']
    opening_sha = hashlib.sha256(json.dumps(arena['openings'], separators=(',', ':')).encode()).hexdigest()
    assert opening_sha == opening_record['openings_sha256']
    arrays, manifest = read_data(data_directory)
    assert manifest['npz_sha256'] == arena['data_sha256'] == training['data_sha256']
    for filename, expected_sha in arena['models'].items():
        assert filename in ('initial-policy.json', 'best-policy.json', 'fixed-v2-policy.json')
        assert sha256(directory / filename) == expected_sha
    if 'fixed-v2-policy.json' in arena['models']:
        assert weights_digest(load_network(directory / 'fixed-v2-policy.json')) == (
            '922df69128314e622bfbca469bba3fb19867adedccb8ae9d9d8890fb489f7abb')
    torch.set_num_threads(1)
    validation = np.flatnonzero(arrays['split'] == 'validation')
    model_results = {}
    for name, filename, metric_key, digest_key in (
        ('initial', 'initial-policy.json', 'initial_metrics', 'initial_weights_sha256'),
        ('selected', 'best-policy.json', 'selected_metrics', 'selected_weights_sha256'),
    ):
        path = directory / filename
        assert sha256(path) == arena['models'][filename]
        model = load_network(path)
        assert weights_digest(model) == training[digest_key]
        totals = np.zeros(3)
        with torch.no_grad():
            for offset in range(0, len(validation), 128):
                ids = validation[offset:offset + 128]
                logits, value = model(torch.from_numpy(arrays['x'][ids]))
                logits = logits.masked_fill(~torch.from_numpy(arrays['legal'][ids]), -1e9)
                labels = torch.from_numpy(arrays['policy'][ids])
                ce = torch.nn.functional.cross_entropy(logits, labels)
                mse = torch.nn.functional.mse_loss(value, torch.from_numpy(arrays['value'][ids]))
                accuracy = (logits.argmax(1) == labels).float().mean()
                totals += np.array([float(ce), float(mse), float(accuracy)]) * len(ids)
        totals /= len(validation)
        measured = dict(zip(('policy_ce', 'value_mse', 'policy_accuracy'), totals.tolist()))
        measured['joint_loss'] = measured['policy_ce'] + measured['value_mse']
        error = max(abs(v - training[metric_key]['validation'][k]) for k, v in measured.items())
        assert error < 1e-5
        model_results[name] = {'validation': measured, 'max_cpu_cuda_difference': error,
                               'weights_sha256': training[digest_key], 'file_sha256': sha256(path)}

    seen, groups, reasons, plies = set(), {}, Counter(), 0
    anomalies = Counter()
    for row in arena['games']:
        assert row['game_id'] not in seen
        seen.add(row['game_id'])
        assert row['candidate_color'] in (0, 1)
        prefix = arena['openings'][row['pair_id']]
        assert row['opening'] == prefix and row['moves'][:len(prefix)] == prefix
        groups.setdefault((row['opponent'], row['pair_id']), []).append(row['candidate_color'])
        game = Game()
        for usi in row['moves']:
            assert not game.result
            game.push(usi)
            plies += 1
        if row['reason'] in ('詰み', '合法手なし', '千日手', '連続王手の千日手'):
            assert game.result and game.result['reason'] == row['reason']
            expected = .5 if game.result['winner'] is None else float(game.result['winner'] == row['candidate_color'])
            assert expected == row['score']
        elif row['reason'] == 'move_limit':
            assert not game.result and row['plies'] == arena['config']['max_plies']
        else:
            # Time, illegal move, teacher resignation, and other exceptional
            # endings are reported, never silently recast as normal games.
            anomalies[row['reason']] += 1
        reasons[row['reason']] += 1
        for key in ('illegal_move', 'time_forfeit', 'abnormal'):
            anomalies[key] += int(row[key])
    assert len(seen) == len(arena['comparisons']) * arena['config']['pairs'] * 2
    assert len(groups) == len(arena['comparisons']) * arena['config']['pairs']
    assert all(sorted(colors) == [0, 1] for colors in groups.values())
    result = {'status': 'passed', 'arena_source': arena['source'], 'games': len(seen),
              'legal_plies_replayed': plies, 'pairing_verified': True,
              'openings_sha256': opening_sha, 'reasons': dict(reasons), 'anomalies': dict(anomalies),
              'validation_only': True, 'test_metrics_computed': False, 'models': model_results,
              'arena_report_sha256': sha256(directory / 'policy-arena-report.json')}
    write_json(directory / 'evaluation-audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--opening-commitment', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.data_dir, args.opening_commitment), ensure_ascii=False, indent=2))
