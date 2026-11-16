"""Optional cooperative checking-mate proof before policy search.

Only a completed proof over every legal defense can override PUCT. Unknown,
interrupted or depth-limited results never count as a mate. This is an opt-in
experiment; the ordinary policy search and deployed configurations are unchanged.
"""
from ..board import new_board,clone_board,require_rule_board,push_legal_usi,is_rule_board
import copy
from dataclasses import dataclass
import math
import time

import shogi

from ..engine import MATE, SearchResult, SearchStopped
from .search import choose_policy_move


@dataclass(frozen=True)
class MateResult:
    move: shogi.Move | None
    plies: int
    nodes: int
    status: str
    certificate: dict | None = None


@dataclass(frozen=True)
class TacticalResult:
    search: SearchResult
    mate: MateResult
    elapsed_seconds: float


def find_checking_mate(board, *, seconds=.1, max_plies=5, max_nodes=2000, stop=None):
    """Prove a 1/3/5-ply checking mate, retaining repetition history.

    A repetition is never accepted as a mate. The budget is checked between
    legal moves and recursive nodes; an individual library call is cooperative.
    """
    require_rule_board(board)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if max_plies not in (1, 3, 5):
        raise ValueError('max_plies must be 1, 3 or 5')
    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')
    deadline = time.monotonic() + seconds
    position = clone_board(board)
    attacker = position.turn
    nodes = 0
    root_checks = []

    def check():
        if stop is not None and stop.is_set():
            raise SearchStopped('cancelled')
        if nodes >= max_nodes or time.monotonic() >= deadline:
            raise SearchStopped('budget_exhausted')

    def prepare_root_checks():
        # Most forcing checks first. Count up to three legal replies solely
        # for ordering; proof still examines EVERY defense recursively.
        for move in position.legal_moves:
            check()
            position.push(move)
            try:
                if not position.is_check():
                    continue
                replies = 0
                for _ in position.legal_moves:
                    check()
                    replies += 1
                    if replies == 3:
                        break
                if replies == 0 and not position.is_fourfold_repetition():
                    # The checking move has no legal defense: its proof is
                    # complete without inspecting unrelated root candidates.
                    return move
                root_checks.append((replies, move))
            finally:
                position.pop()
        root_checks.sort(key=lambda item: item[0])

    def solve(remaining, ply=0):
        nonlocal nodes
        check()
        nodes += 1
        if position.is_fourfold_repetition():
            return None
        if position.turn != attacker:
            defenses = {}
            # Every legal defense must be disproved, including captures,
            # interpositions and drops. No network prior may prune them.
            for move in position.legal_moves:
                check()
                if remaining == 0:
                    return None
                position.push(move)
                try:
                    result = solve(remaining-1, ply+1)
                finally:
                    position.pop()
                if result is None:
                    return None
                defenses[move.usi()] = result
            if not defenses:
                return (0, {'mate': True}) if position.is_check() else None
            return 1 + max(value[0] for value in defenses.values()), {
                'defenses': {move: value[1] for move, value in defenses.items()}}
        if remaining == 0:
            return None
        moves = (item[1] for item in root_checks) if ply == 0 else position.legal_moves
        for move in moves:
            check()
            position.push(move)
            try:
                if not position.is_check():
                    continue
                result = solve(remaining-1, ply+1)
                if result is not None:
                    return result[0]+1, {'attack': move.usi(), 'continuation': result[1]}
            finally:
                position.pop()
        return None

    try:
        check()
        if position.is_fourfold_repetition():
            return MateResult(None, 0, nodes, 'repetition')
        immediate = prepare_root_checks()
        if immediate is not None:
            nodes += 1
            check()  # Honor cancellation/deadline/node gates before commit.
            return MateResult(immediate, 1, nodes, 'proved',
                              {'attack': immediate.usi(), 'continuation': {'mate': True}})
        for depth in range(1, max_plies+1, 2):
            result = solve(depth)
            if result is not None:
                return MateResult(shogi.Move.from_usi(result[1]['attack']), result[0], nodes,
                                  'proved', result[1])
    except SearchStopped as exc:
        return MateResult(None, 0, nodes, str(exc))
    return MateResult(None, 0, nodes, 'no_checking_mate_within_bound')


def choose_tactical_move(board, evaluator, *, seconds=.1, max_simulations=128,
                         mate_fraction=.2, mate_plies=5, mate_nodes=2000, stop=None):
    """Spend part of the SAME total clock on proof, then use remaining PUCT time.

    Proof nodes and PUCT simulations are reported separately. A proved mate
    returns zero policy simulations rather than misreporting them as NN search.
    """
    require_rule_board(board)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if not math.isfinite(mate_fraction) or not 0 <= mate_fraction <= 1:
        raise ValueError('mate_fraction must be within [0,1]')
    if not isinstance(max_simulations, int) or max_simulations < 1:
        raise ValueError('max_simulations must be a positive integer')
    started = time.monotonic()
    proof = find_checking_mate(board, seconds=seconds*mate_fraction, max_plies=mate_plies,
                               max_nodes=mate_nodes, stop=stop)
    if proof.move is not None:
        result = SearchResult(proof.move, 0, 0, MATE-proof.plies)
    else:
        remaining = max(0, seconds-(time.monotonic()-started))
        result = choose_policy_move(board, evaluator, seconds=remaining,
                                    max_simulations=max_simulations, stop=stop)
    return TacticalResult(result, proof, time.monotonic()-started)
