"""Self-generated positions, supervised search labels, and game-level splits."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from multiprocessing import get_context
from pathlib import Path
import random

import shogi

from ..engine import choose_move, evaluate, ordered_moves
from ..game import Game
from .common import BudgetExpired, sha256, source_identity, write_json
from .features import CORRECTION_CAP, SCALE, encode, position_key


def split_for_game(game_id):
    return 'test' if game_id % 10 == 0 else 'validation' if game_id % 10 == 1 else 'train'


def label_game(task):
    game_id, settings = task
    rng = random.Random(settings['data_seed'] + game_id)
    game = Game()
    selected = set(rng.sample(range(16, settings['data_max_plies']), settings['samples_per_game']))
    positions = []
    for ply in range(settings['data_max_plies']):
        if game.result:
            break
        if ply in selected:
            positions.append((ply, game.board.sfen()))
        moves = ordered_moves(game.board)
        # Diverse, reproducible trajectories; this policy is not the teacher.
        move = rng.choice(moves if rng.random() < 0.3 else moves[:min(4, len(moves))])
        game.push(move.usi())
    rows = []
    for ply, sfen in positions:
        board = shogi.Board(sfen)
        result = choose_move(board, seconds=2.0, max_depth=2, max_nodes=settings['teacher_nodes'])
        if result.depth < settings['teacher_min_depth'] or abs(result.score) >= 90_000:
            continue
        static = evaluate(board)
        rows.append({'game_id': game_id, 'split': split_for_game(game_id), 'ply': ply, 'sfen': sfen,
                     'static_cp': static, 'teacher_cp': result.score,
                     'target': max(-CORRECTION_CAP, min(CORRECTION_CAP, result.score - static)) / SCALE,
                     'teacher_depth': result.depth, 'teacher_nodes': result.nodes})
    return {'game_id': game_id, 'rows': rows}


def feature_key(vector):
    """Also deduplicate mirror-equivalent inputs before any augmentation."""
    sparse = tuple((i, value) for i, value in enumerate(vector) if value)
    mirror = tuple(sorted(((i // 81 * 81 + i % 81 // 9 * 9 + 8 - i % 9) if i < 1134 else i, value)
                          for i, value in sparse))
    return min(sparse, mirror)


def prepare_data(directory, config, budget):
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {key: config[key] for key in ('data_seed', 'data_games', 'data_max_plies', 'samples_per_game', 'teacher_nodes', 'teacher_min_depth')}
    metadata_path = directory / 'data-settings.json'
    identity = source_identity()
    manifest = {'settings': settings, 'source_sha256': identity['source_sha256']}
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != manifest:
        raise ValueError('Data settings or source changed; use a new run directory.')
    write_json(metadata_path, manifest)
    progress_path = directory / 'data-progress.jsonl'
    completed = {}
    if progress_path.exists():
        lines = progress_path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
                break  # A process may have been interrupted during the final append.
            completed[entry['game_id']] = entry
        progress_path.write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n'
                                         for _, item in sorted(completed.items())))
    pending_ids = iter(i for i in range(settings['data_games']) if i not in completed)
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool, progress_path.open('a') as stream:
        jobs = {}
        exhausted = False
        while jobs or not exhausted:
            budget.check(reserve=30)
            while len(jobs) < config['workers'] and not exhausted:
                try:
                    game_id = next(pending_ids)
                except StopIteration:
                    exhausted = True
                    break
                jobs[pool.submit(label_game, (game_id, settings))] = game_id
            if not jobs:
                break
            done, _ = wait(jobs, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                entry = future.result()
                completed[entry['game_id']] = entry
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                stream.flush()
                del jobs[future]
                budget.save()
                if len(completed) % 10 == 0:
                    print(f'data: {len(completed)}/{settings["data_games"]} games', flush=True)
    seen, rows, vectors = set(), [], []
    for _, entry in sorted(completed.items()):
        for row in entry['rows']:
            vector = encode(shogi.Board(row['sfen']))
            key = feature_key(vector)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            vectors.append(vector)
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'validation', 'test')}
    if min(counts.values()) < 4:
        raise ValueError(f'Too few disjoint examples: {counts}; increase data_games in a new run.')
    corpus = directory / 'corpus.jsonl'
    corpus.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    arrays = {'x': np.asarray(vectors, dtype=np.float32),
              'y': np.asarray([row['target'] for row in rows], dtype=np.float32),
              'split': np.asarray([row['split'] for row in rows]),
              'game_id': np.asarray([row['game_id'] for row in rows], dtype=np.int64)}
    np.savez_compressed(directory / 'data.npz', **arrays)
    report = {'provenance': 'positions generated by this repository; no external games, datasets or weights',
              'teacher': 'frozen handcrafted material + alpha-beta, max depth 2, node bounded',
              'settings': settings, 'counts': counts, 'games': len(completed),
              'corpus_sha256': sha256(corpus), 'source': identity,
              'teacher_depth_counts': {str(depth): sum(row['teacher_depth'] == depth for row in rows) for depth in (1, 2)},
              'split_policy': 'whole games; exact and mirrored feature duplicates removed across all splits'}
    write_json(directory / 'data-manifest.json', report)
    return arrays, report
