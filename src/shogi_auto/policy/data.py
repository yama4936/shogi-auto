"""CPU-only supervised labels from a pinned USI teacher; no external games."""
import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import random
import time

import shogi

from ..game import Game
from ..learning.data import feature_key as old_feature_key, split_for_game
from ..learning.features import encode as old_encode
from .common import Budget, BudgetExpired, sha256, source_identity, write_json
from .features import ACTIONS, action_index, encode, feature_key, legal_actions
from .teacher import UsiTeacher


def generate(directory, binary, eval_dir, old_datasets, config, max_seconds=1800):
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    forbidden, forbidden_policy = set(), set()
    prior_data = {}
    for raw_path in old_datasets:
        path = Path(raw_path)
        prior_data[path.name + ':' + sha256(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as previous:
            vectors = previous['x']
            if vectors.ndim == 2 and vectors.shape[1] == 1141:
                for vector in vectors:forbidden.add(old_feature_key(vector))
            elif vectors.ndim == 4 and vectors.shape[1:] == (43,9,9):
                for x in vectors:
                    forbidden_policy.add(min(hashlib.sha256(x[:42].tobytes()).hexdigest(),
                                             hashlib.sha256(x[:42,:,::-1].copy().tobytes()).hexdigest()))
            else:raise ValueError('Unsupported previous dataset representation')
    identity = {'source': source_identity(), 'config': config,
                'features_sha256': sha256(Path(__file__).with_name('features.py')),
                'teacher_binary_sha256': sha256(binary), 'teacher_weights_sha256': sha256(Path(eval_dir) / 'nn.bin'),
                'previous_datasets': prior_data, 'value_target': 'tanh(side-to-move teacher cp / 600); mate signed 1',
                'external_game_records': False, 'teacher_weight_transfer': False}
    path = directory / 'data-settings.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Data source or settings changed; use a new directory')
    write_json(path, identity)
    progress = directory / 'data-progress.jsonl'
    completed = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            item = json.loads(line)
            completed[item['game_id']] = item
    budget = Budget(directory, max_seconds, visible_gpus=0)
    try:
        with UsiTeacher(binary, eval_dir, directory / 'teacher-usi.log') as teacher, progress.open('a') as stream:
            for game_id in range(config['games']):
                if game_id in completed:
                    continue
                budget.check(reserve=30)
                teacher.new_game()
                rng = random.Random(config['seed'] + game_id)
                selected = set(rng.sample(range(16, config['max_plies']), config['samples_per_game']))
                game, rows, reasons, available = Game(), [], Counter(), 0
                for ply in range(config['max_plies']):
                    if game.result:
                        break
                    budget.check(reserve=10)
                    available += int(ply in selected)
                    query = teacher.analyze(game.board, nodes=config['trajectory_nodes'])
                    if query['move'] is None:
                        reasons['teacher_terminal_declaration'] += 1
                        break
                    if ply in selected:
                        if (old_feature_key(old_encode(game.board)) in forbidden or
                                feature_key(game.board) in forbidden_policy):
                            reasons['overlap_previous_experiments'] += 1
                        else:
                            label = teacher.analyze(game.board, nodes=config['teacher_nodes'])
                            if label['move'] is None:
                                reasons['no_legal_policy_label'] += 1
                            else:
                                value = (math.copysign(1, label['score']) if label['score_kind'] == 'mate'
                                         else math.tanh(label['score'] / 600))
                                rows.append({'game_id': game_id, 'split': split_for_game(game_id),
                                             'sfen': game.board.sfen(), 'moves': [m.usi() for m in game.board.move_stack],
                                             'policy_usi': label['usi'], 'value': value,
                                             'teacher': {k: v for k, v in label.items() if k != 'move'},
                                             'repetitions': game.board.transpositions[game.board.zobrist_hash()]})
                    # Seeded diversity remains separate from the teacher label.
                    choices = [m for m in game.board.legal_moves]
                    if ply < 12 or rng.random() < .08:
                        move = rng.choice(choices)
                    else:
                        move = query['move']
                    game.push(move.usi())
                item = {'game_id': game_id, 'requested': len(selected), 'available': available,
                        'rows': rows, 'rejections': dict(reasons), 'trajectory_plies': len(game.history),
                        'trajectory_result': game.result}
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                stream.flush()
                completed[game_id] = item
                budget.save()
                if (game_id + 1) % 4 == 0:
                    print(f'policy data: {game_id + 1}/{config["games"]} games', flush=True)
    finally:
        budget.save()
    seen, rows, xs, masks, policies, values = set(), [], [], [], [], []
    counts = Counter()
    for game_id in sorted(completed):
        for name, amount in completed[game_id]['rejections'].items():
            counts[name] += amount
        for row in completed[game_id]['rows']:
            board = shogi.Board()
            for usi in row['moves']:
                board.push_usi(usi)
            assert board.sfen() == row['sfen']
            key = feature_key(board)
            if key in seen:
                counts['duplicate_or_mirrored'] += 1
                continue
            seen.add(key)
            actions = legal_actions(board)
            target = action_index(board, shogi.Move.from_usi(row['policy_usi']))
            if target not in actions:
                raise ValueError('Invalid teacher policy target')
            mask = np.zeros(ACTIONS, dtype=np.bool_)
            mask[list(actions)] = True
            xs.append(encode(board));masks.append(mask);policies.append(target);values.append(row['value']);rows.append(row)
    corpus = directory / 'corpus.jsonl'
    corpus.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    arrays = {'x': np.asarray(xs, dtype=np.float32), 'legal': np.asarray(masks),
              'policy': np.asarray(policies, dtype=np.int64), 'value': np.asarray(values, dtype=np.float32),
              'split': np.asarray([row['split'] for row in rows]),
              'game_id': np.asarray([row['game_id'] for row in rows], dtype=np.int64)}
    split_counts = {name: int((arrays['split'] == name).sum()) for name in ('train', 'validation', 'test')}
    if min(split_counts.values()) < config['min_split_rows']:
        raise ValueError(f'Too few independent data: {split_counts}')
    np.savez_compressed(directory / 'data.npz', **arrays)
    manifest = {**identity, 'counts': split_counts, 'rows': len(rows), 'rejections': dict(counts),
                'sampling_counts': {'requested': sum(v['requested'] for v in completed.values()),
                                    'available': sum(v['available'] for v in completed.values()),
                                    'labelled_before_dedup': sum(len(v['rows']) for v in completed.values()),
                                    'selected_not_reached': sum(v['requested']-v['available'] for v in completed.values())},
                'games': len(completed), 'corpus_sha256': sha256(corpus), 'npz_sha256': sha256(directory / 'data.npz'),
                'teacher_identity': teacher.identity, 'budget': json.loads(budget.path.read_text()),
                'split_policy': 'whole games; canonical/mirrored states deduplicated globally; prior v1 inputs excluded'}
    write_json(directory / 'data-manifest.json', manifest)
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--teacher-binary', type=Path, required=True)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--old-data', type=Path, action='append', required=True)
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--samples-per-game', type=int, default=16)
    parser.add_argument('--max-plies', type=int, default=160)
    parser.add_argument('--teacher-nodes', type=int, default=32768)
    parser.add_argument('--trajectory-nodes', type=int, default=2048)
    parser.add_argument('--min-split-rows', type=int, default=100)
    parser.add_argument('--max-wall-seconds', type=float, default=1800)
    parser.add_argument('--seed', type=int, default=20261104)
    args = parser.parse_args()
    if not (20 <= args.games <= 1024 and 32 <= args.max_plies <= 256 and
            1 <= args.samples_per_game <= min(32, args.max_plies - 16) and
            1024 <= args.teacher_nodes <= 65536 and 256 <= args.trajectory_nodes <= 8192):
        raise ValueError('Data generation settings exceed bounded CPU experiment limits')
    config = {key: getattr(args, key) for key in ('games', 'samples_per_game', 'max_plies', 'teacher_nodes',
                                                'trajectory_nodes', 'min_split_rows', 'seed')}
    _, report = generate(args.run_dir, args.teacher_binary, args.eval_dir, args.old_data, config, args.max_wall_seconds)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
