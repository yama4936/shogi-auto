"""Paired held-out openings, equal budgets, outcomes and cluster uncertainty."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import math
from multiprocessing import get_context
from pathlib import Path
import random
import time

import shogi

from ..engine import choose_move, evaluate
from ..game import Game
from .common import sha256, source_identity, write_json
from .data import feature_key
from .features import LinearEvaluator, encode


def paired_interval(pair_scores, seed=7391, samples=4000):
    """Percentile bootstrap over opening pairs (never pretend games are IID)."""
    if not pair_scores:
        return [0.0, 1.0]
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(pair_scores, k=len(pair_scores))) / len(pair_scores)
                       for _ in range(samples))
    n = len(estimates)
    return [estimates[int(n * 0.025)], estimates[min(n - 1, int(n * 0.975))]]


def summarize(games, expected_pairs):
    from collections import Counter, defaultdict
    outcomes = Counter(game['outcome'] for game in games)
    reasons = Counter(game['reason'] for game in games)
    pairs = defaultdict(list)
    for game in games:
        pairs[game['pair_id']].append(game['score'])
    complete_pairs = [sum(scores) / 2 for scores in pairs.values() if len(scores) == 2]
    interval = paired_interval(complete_pairs)
    mean_pair_score = sum(complete_pairs) / len(complete_pairs) if complete_pairs else None
    radius = math.sqrt(math.log(40) / (2 * len(complete_pairs))) if complete_pairs else 1
    conservative = ([max(0.0, mean_pair_score - radius), min(1.0, mean_pair_score + radius)]
                    if complete_pairs else [0.0, 1.0])
    abnormal = sum(game['abnormal'] for game in games)
    move_limits = reasons.get('move_limit', 0)
    complete = len(complete_pairs) == expected_pairs and len(games) == expected_pairs * 2
    return {'games': len(games), 'complete_pairs': len(complete_pairs), 'expected_pairs': expected_pairs,
            'wins': outcomes.get('win', 0), 'losses': outcomes.get('loss', 0), 'draws': outcomes.get('draw', 0),
            'mean_score': sum(game['score'] for game in games) / len(games) if games else None,
            'score_95pct_pair_bootstrap': interval, 'reason_counts': dict(reasons),
            'score_95pct_hoeffding_bound': conservative,
            'illegal_moves': sum(game['illegal_move'] for game in games),
            'time_forfeits': sum(game['time_forfeit'] for game in games), 'abnormal_terminations': abnormal,
            'move_limit_games': move_limits, 'complete': complete,
            'improvement_evidence': complete and len(complete_pairs) >= 30 and conservative[0] > 0.5 and
                                    abnormal == 0 and move_limits / max(1, len(games)) <= 0.2,
            'warning': 'Small samples or many move-limit games cannot establish strength improvement. No Elo or rank is inferred.'}


def create_opening(seed, forbidden):
    rng = random.Random(seed)
    for _ in range(100):
        game = Game()
        prefix = []
        for _ in range(12):
            moves = list(game.board.legal_moves)
            if not moves or game.result:
                break
            move = rng.choice(moves)
            game.push(move.usi())
            prefix.append(move.usi())
        if (not game.result and not game.board.is_check() and abs(evaluate(game.board)) <= 200 and
                feature_key(encode(game.board)) not in forbidden):
            return prefix
    raise ValueError('Could not generate a disjoint balanced opening')


def play_game(task):
    game_id, opponent, pair_id, color, prefix, candidate_path, initial_path, config = task
    candidate = LinearEvaluator.load(candidate_path)
    opponent_eval = None if opponent == 'frozen-material' else LinearEvaluator.load(initial_path)
    game = Game()
    for move in prefix:
        game.push(move)
    timings, depth_counts, node_counts = [[], []], [[], []], [[], []]
    root_coverage, fallback_turns = [[], []], [0, 0]
    error, time_forfeit, illegal = None, False, False
    winner, reason = None, 'move_limit'
    hard_limit = max(0.5, config['move_seconds'] * 5)
    for _ in range(config['arena_max_plies']):
        if game.result:
            winner, reason = game.result['winner'], game.result['reason']
            break
        mover = game.board.turn
        is_candidate = mover == color
        evaluator = candidate if is_candidate else opponent_eval
        side = 0 if is_candidate else 1
        started = time.monotonic()
        try:
            result = choose_move(game.board, seconds=config['move_seconds'], max_nodes=6000, evaluator=evaluator)
            elapsed = time.monotonic() - started
            timings[side].append(elapsed)
            depth_counts[side].append(result.depth)
            node_counts[side].append(result.nodes)
            root_coverage[side].append(result.root_moves_evaluated / max(1, result.root_legal_moves))
            fallback_turns[side] += int(result.depth == 0 and result.root_moves_evaluated == 0)
            if elapsed > hard_limit:
                time_forfeit, winner, reason = True, 1 - mover, 'time_forfeit'
                break
            if result.move is None or result.move not in game.board.legal_moves:
                illegal, winner, reason = True, 1 - mover, 'illegal_move'
                break
            game.push(result.move.usi())
        except Exception as exc:
            error, winner, reason = type(exc).__name__ + ': ' + str(exc), 1 - mover, 'exception'
            break
    if game.result:
        winner, reason = game.result['winner'], game.result['reason']
    outcome = 'draw' if winner is None else 'win' if winner == color else 'loss'
    return {'game_id': game_id, 'opponent': opponent, 'pair_id': pair_id, 'candidate_color': color,
            'opening': prefix, 'outcome': outcome, 'score': 0.5 if winner is None else float(winner == color),
            'reason': reason, 'error': error, 'abnormal': bool(error or illegal or time_forfeit),
            'time_forfeit': time_forfeit, 'illegal_move': illegal,
            'moves': [row['usi'] for row in game.history], 'plies': len(game.history) - len(prefix),
            'timing_seconds': [{'mean': sum(values) / len(values) if values else None,
                                'max': max(values) if values else None} for values in timings],
            'mean_depth': [sum(values) / len(values) if values else None for values in depth_counts],
            'zero_evaluated_fallback_turns': fallback_turns,
            'mean_last_iteration_root_coverage': [sum(values) / len(values) if values else None for values in root_coverage],
            'total_nodes': [sum(values) for values in node_counts]}


def evaluate_models(directory, arrays, config, budget):
    directory = Path(directory)
    candidate_path = directory / 'best-model.json'
    initial_path = directory / 'initial-model.json'
    identity = {'source': source_identity(), 'candidate_sha256': sha256(candidate_path),
                'initial_sha256': sha256(initial_path),
                'settings': {key: config[key] for key in ('arena_seed', 'arena_pairs', 'move_seconds', 'arena_max_plies')}}
    identity_path = directory / 'arena-settings.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Arena settings, models or source changed; use a new run directory.')
    write_json(identity_path, identity)
    forbidden = {feature_key(vector) for vector in arrays['x']}
    openings = [create_opening(config['arena_seed'] + i, forbidden) for i in range(config['arena_pairs'])]
    progress_path = directory / 'arena-progress.jsonl'
    completed = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            completed[entry['game_id']] = entry
        progress_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for _, row in sorted(completed.items())))
    tasks = []
    for opponent in ('frozen-material', 'initial-model'):
        for pair_id, prefix in enumerate(openings):
            for color in (shogi.BLACK, shogi.WHITE):
                game_id = f'{opponent}:{pair_id}:{color}'
                if game_id not in completed:
                    tasks.append((game_id, opponent, pair_id, color, prefix, str(candidate_path), str(initial_path), config))
    pending, exhausted = iter(tasks), False
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool, progress_path.open('a') as stream:
        jobs = {}
        while jobs or not exhausted:
            budget.check(reserve=60)
            while len(jobs) < config['workers'] and not exhausted:
                try:
                    task = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                jobs[pool.submit(play_game, task)] = task[0]
            if not jobs:
                break
            done, _ = wait(jobs, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                completed[row['game_id']] = row
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                del jobs[future]
                budget.save()
                print(f'arena: {len(completed)}/{config["arena_pairs"] * 4} {row["game_id"]} {row["outcome"]} ({row["reason"]})', flush=True)
    games = [row for _, row in sorted(completed.items())]
    summaries = {opponent: summarize([row for row in games if row['opponent'] == opponent], config['arena_pairs'])
                 for opponent in ('frozen-material', 'initial-model')}
    report = {**identity, 'comparisons': summaries, 'games': games,
              'controls': 'same 12-ply held-out openings, colors swapped, same node cap and nominal per-move budget',
              'hard_move_limit_seconds': max(0.5, config['move_seconds'] * 5),
              'rules_limitations': 'laboratory games: move-limit draws; no entering-king declarations, no CSA clock semantics',
              'complete': all(row['complete'] for row in summaries.values()),
              'improvement_evidence': all(row['improvement_evidence'] for row in summaries.values())}
    write_json(directory / 'arena-report.json', report)
    return report
