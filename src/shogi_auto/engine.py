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


def ordered_moves(board, preferred=None):
    def priority(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square) if move.from_square is not None else None
        return (100_000 if move == preferred else 0) + (
            VALUES[victim.piece_type] * 10 - (VALUES[attacker.piece_type] if attacker else 0)
            if victim else 0
        ) + (500 if move.promotion else 0)
    return sorted(board.legal_moves, key=priority, reverse=True)


class SearchStopped(Exception):
    pass


def choose_move(board, seconds=0.35, max_depth=3, max_nodes=6000, stop=None):
    """Search an isolated copy and always return a legal fallback on timeout.

    The deadline is cooperative at node boundaries (not a hard real-time limit).
    Repetition in search is scored as a draw; the game adjudicates checking cycles.
    """
    return _search(copy.deepcopy(board), seconds, max_depth, max_nodes, stop)


def _search(board, seconds, max_depth, max_nodes, stop):
    stop = stop if stop is not None else threading.Event()
    deadline = time.monotonic() + max(0.001, min(seconds, 2.0))
    nodes = 0
    moves = ordered_moves(board)
    if not moves:
        return SearchResult(None, 0, 0, -MATE)
    best = SearchResult(moves[0], 0, 0, evaluate(board))

    def check_budget():
        if nodes >= max_nodes or stop.is_set() or time.monotonic() >= deadline:
            raise SearchStopped

    def negamax(depth, alpha, beta, ply):
        nonlocal nodes
        check_budget()
        nodes += 1
        if board.is_fourfold_repetition():
            return 0
        legal = ordered_moves(board)
        if not legal:
            return -MATE + ply
        if depth == 0:
            return evaluate(board)
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

    for depth in range(1, max_depth + 1):
        candidate, score = best.move, -MATE - 1
        try:
            for move in ordered_moves(board, best.move):
                check_budget()
                board.push(move)
                try:
                    value = -negamax(depth - 1, -MATE, -score, 1)
                finally:
                    board.pop()
                if value > score:
                    candidate, score = move, value
        except SearchStopped:
            break
        best = SearchResult(candidate, depth, nodes, score)
        if score >= MATE - 100:
            break
    return SearchResult(best.move, best.depth, nodes, best.score)
