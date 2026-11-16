"""Small CPU baseline: iterative deepening, alpha-beta and material evaluation."""
from dataclasses import dataclass
import copy
import threading
import time

import shogi

VALUES = (0, 100, 300, 320, 450, 550, 850, 1000, 0, 550, 550, 550, 550, 1050, 1250)
MATE = 100_000


@dataclass(frozen=True)
class SearchResult:
    move: shogi.Move | None
    depth: int
    nodes: int
    score: int
    root_moves_evaluated: int = 0
    root_legal_moves: int = 0
    attempted_depth: int = 0


def evaluate(board):
    """Material, hands, and a small advancement bonus, from side-to-move."""
    score = 0
    for square in shogi.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = square // 9
            advance = 8 - rank if piece.color == shogi.BLACK else rank
            value = VALUES[piece.piece_type]
            if piece.piece_type in (shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER):
                value += advance * 3
            score += value if piece.color == shogi.BLACK else -value
    for color in (shogi.BLACK, shogi.WHITE):
        value = sum(VALUES[kind] * count for kind, count in board.pieces_in_hand[color].items())
        score += value if color == shogi.BLACK else -value
    return score if board.turn == shogi.BLACK else -score


def ordered_moves(board, preferred=None, legal=None):
    def priority(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square) if move.from_square is not None else None
        return (100_000 if move == preferred else 0) + (
            VALUES[victim.piece_type] * 10 - (VALUES[attacker.piece_type] if attacker else 0)
            if victim else 0
        ) + (500 if move.promotion else 0)
    # LegalMoveGenerator.__len__ itself generates all legal moves. Giving it
    # directly to sorted() triggers a second pass via Python's length hint.
    source = (move for move in board.legal_moves) if legal is None else legal
    return sorted(source, key=priority, reverse=True)


class SearchStopped(Exception):
    pass


def choose_move(board, seconds=0.35, max_depth=3, max_nodes=6000, stop=None, evaluator=None):
    """Search an isolated copy and always return a legal fallback on timeout.

    The deadline is cooperative at node boundaries (not a hard real-time limit).
    Repetition in search is scored as a draw; the game adjudicates checking cycles.
    """
    return _search(copy.deepcopy(board), seconds, max_depth, max_nodes, stop, evaluator or evaluate)


def _search(board, seconds, max_depth, max_nodes, stop, evaluator):
    stop = stop if stop is not None else threading.Event()
    deadline = time.monotonic() + max(0.001, min(seconds, 2.0))
    nodes = 0
    moves = ordered_moves(board)
    if not moves:
        return SearchResult(None, 0, 0, -MATE)
    best = SearchResult(moves[0], 0, 0, evaluator(board))

    def check_budget():
        if nodes >= max_nodes or stop.is_set() or time.monotonic() >= deadline:
            raise SearchStopped

    def negamax(depth, alpha, beta, ply):
        nonlocal nodes
        check_budget()
        nodes += 1
        if board.is_fourfold_repetition():
            return 0
        if depth == 0:
            # A leaf needs to know whether at least one legal reply exists;
            # sorting every possible reply here is a needless extra branching
            # factor. The library restores its board before yielding a move.
            if next(iter(board.legal_moves), None) is None:
                return -MATE + ply
            return evaluator(board)
        legal = ordered_moves(board)
        if not legal:
            return -MATE + ply
        value = -MATE
        for move in legal:
            board.push(move)
            try:
                value = max(value, -negamax(depth - 1, -beta, -alpha, ply + 1))
            finally:
                board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    evaluated, attempted_depth = 0, 0
    for depth in range(1, max_depth + 1):
        evaluated, attempted_depth = 0, depth
        candidate, score = best.move, -MATE - 1
        try:
            for move in ordered_moves(board, best.move, moves):
                check_budget()
                board.push(move)
                try:
                    value = -negamax(depth - 1, -MATE, -score, 1)
                finally:
                    board.pop()
                if value > score:
                    candidate, score = move, value
                evaluated += 1
        except SearchStopped:
            # Before the first iteration completes, retain the best fully
            # evaluated root move rather than reverting to an unevaluated
            # ordering fallback. Report depth 0 honestly, with root coverage.
            if best.depth == 0 and evaluated:
                best = SearchResult(candidate, 0, nodes, score)
            break
        best = SearchResult(candidate, depth, nodes, score)
        if score >= MATE - 100:
            break
    return SearchResult(best.move, best.depth, nodes, best.score, evaluated, len(moves), attempted_depth)
