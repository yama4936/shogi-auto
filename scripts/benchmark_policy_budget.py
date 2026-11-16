"""Local timing diagnostic with a material proxy, not a neural strength arena.

Compare reviewed legacy search with current search using identical fixtures and
synthetic evaluator delays. No model, GPU, teacher, or game server is used.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import shogi

from shogi_auto.engine import evaluate
from shogi_auto.policy.common import source_identity
from shogi_auto.policy.search import choose_policy_move

BASELINE = '38bf45ff08ea030bf6c0c6b4a39eb68443045511'
FIXTURES = {
    'start': shogi.STARTING_SFEN,
    'capture': '4k4/9/9/4r4/4R4/9/9/9/4K4 b - 1',
    'many_drops': '4k4/9/9/9/9/9/9/9/4K4 b RBGSNLP 1',
}


class MaterialProxy:
    def __init__(self, delay):
        self.delay = delay
        self.calls = 0
        self.yields = 0

    def __call__(self, board):
        return self.evaluate_with_budget(board, lambda: None)

    def evaluate_with_budget(self, board, check_budget):
        import math
        self.calls += 1
        moves = []
        check_budget()
        for move in board.legal_moves:
            check_budget()
            if self.delay:
                time.sleep(self.delay)
            self.yields += 1
            check_budget()
            moves.append(move)
        if not moves:
            return {}, -1.0
        return {move: 1 / len(moves) for move in moves}, math.tanh(evaluate(board) / 600)


def legacy_search(ref):
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', '--verify', ref + '^{commit}'], text=True).strip()
    source = subprocess.check_output(
        ['git', '-C', str(root), 'show', commit + ':src/shogi_auto/policy/search.py'], text=True)
    module = types.ModuleType('shogi_auto.policy._timing_baseline')
    sys.modules[module.__name__] = module
    exec(compile(source, '<reviewed legacy search>', 'exec'), module.__dict__)
    return module.choose_policy_move, commit, hashlib.sha256(source.encode()).hexdigest()


def run(ref, repeats, seconds, delay):
    legacy, commit, digest = legacy_search(ref)
    rows, equivalence = [], []
    for name, sfen in FIXTURES.items():
        board = shogi.Board(sfen)
        # With the deadline disabled by a generous budget, selection, value,
        # completed simulations and depth must agree for an identical evaluator.
        old = legacy(board, MaterialProxy(0), seconds=60, max_simulations=16)
        new = choose_policy_move(board, MaterialProxy(0), seconds=60, max_simulations=16)
        same = old == new
        equivalence.append({'fixture': name, 'same_search_result_at_16_simulations': same})
        if not same:
            raise AssertionError(f'Unconstrained search changed: {name}')
        for mode, pause in (('cpu_proxy', 0), ('injected_delay', delay)):
            for repeat in range(repeats):
                engines = [('before', legacy), ('after', choose_policy_move)]
                if repeat % 2:
                    engines.reverse()
                for version, search in engines:
                    before = board.sfen(), list(board.move_stack), +Counter(board.transpositions)
                    proxy = MaterialProxy(pause)
                    started = time.monotonic()
                    result = search(board, proxy, seconds=seconds, max_simulations=128)
                    elapsed = time.monotonic() - started
                    legal = result.move is not None and board.is_legal(result.move)
                    preserved = before == (board.sfen(), list(board.move_stack), +Counter(board.transpositions))
                    if not legal or not preserved:
                        raise AssertionError('Illegal reply or mutated caller board')
                    rows.append({'fixture': name, 'mode': mode, 'repeat': repeat, 'version': version,
                                 'elapsed_seconds': elapsed, 'budget_excess_seconds': max(0, elapsed-seconds),
                                 'over_arena_half_second_threshold': elapsed > .5,
                                 'completed_simulations': result.nodes, 'max_simulated_ply': result.depth,
                                 'evaluator_calls': proxy.calls, 'legal_yields': proxy.yields,
                                 'legal_reply': legal, 'caller_preserved': preserved})
    summaries = []
    for mode in ('cpu_proxy', 'injected_delay'):
        for version in ('before', 'after'):
            group = [row for row in rows if row['mode'] == mode and row['version'] == version]
            elapsed = sorted(row['elapsed_seconds'] for row in group)
            summaries.append({'mode': mode, 'version': version, 'searches': len(group),
                              'max_elapsed_seconds': max(elapsed),
                              'mean_elapsed_seconds': sum(elapsed)/len(elapsed),
                              'over_arena_half_second_threshold': sum(row['over_arena_half_second_threshold'] for row in group),
                              'total_completed_simulations': sum(row['completed_simulations'] for row in group)})
    return {'purpose': __doc__, 'python': platform.python_version(), 'platform': platform.platform(),
            'python_shogi': package_version('python-shogi'),
            'baseline_commit': commit, 'baseline_search_sha256': digest, 'candidate_source': source_identity(),
            'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'config': {'repeats': repeats, 'seconds': seconds, 'delay_per_legal_yield_seconds': delay,
                       'max_simulations': 128, 'fixture_sfens': FIXTURES},
            'limitations': ['Material proxy, no neural model or teacher; not a strength measurement',
                            'Injected delays are synthetic, not a diagnosis of historical time forfeits',
                            'Cooperative checks cannot preempt a blocked evaluator or PyTorch operation',
                            'CPU scheduling and a single legal-generator step may still exceed the deadline'],
            'unconstrained_equivalence': equivalence, 'summaries': summaries, 'searches': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default=BASELINE)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=.05)
    parser.add_argument('--delay', type=float, default=.001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.repeats <= 10 and 0 < args.seconds <= 2 and 0 <= args.delay <= .01):
        parser.error('repeats 1..10, seconds (0,2], delay [0,.01] required')
    if args.output.exists():
        parser.error('Use a new output path to preserve prior evidence')
    report = run(args.baseline_ref, args.repeats, args.seconds, args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summaries'], indent=2))


if __name__ == '__main__':
    main()
