"""Side-to-move planes and legal, injective shogi move labels.

Fresh implementation of the usual 27-direction/drop policy representation.
No third-party trained parameters or game records are embedded.
"""
import hashlib

import shogi

PLANES = 43
ACTIONS = 27 * 81
HAND_LIMITS = (18, 4, 4, 4, 4, 2, 2)
DIRECTIONS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (-1, -2), (1, -2))


def encode(board):
    import numpy as np
    x = np.zeros((PLANES, 9, 9), dtype=np.float32)
    for square in shogi.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            normalized = square if board.turn == shogi.BLACK else 80 - square
            channel = piece.piece_type - 1 + 14 * int(piece.color != board.turn)
            x[channel, normalized // 9, normalized % 9] = 1
    for side in (0, 1):
        color = board.turn if side == 0 else 1 - board.turn
        for kind in range(1, 8):
            x[28 + 7 * side + kind - 1].fill(board.pieces_in_hand[color].get(kind, 0) / HAND_LIMITS[kind - 1])
    # Position transposition count includes the current position.
    x[42].fill(min(3, max(0, board.transpositions[board.zobrist_hash()] - 1)) / 3)
    return x


def action_index(board, move):
    destination = move.to_square if board.turn == shogi.BLACK else 80 - move.to_square
    if move.from_square is None:
        plane = 20 + move.drop_piece_type - 1
    else:
        source = move.from_square if board.turn == shogi.BLACK else 80 - move.from_square
        dy, dx = destination // 9 - source // 9, destination % 9 - source % 9
        direction = ((dx, dy) if abs(dy) == 2 and abs(dx) == 1 else
                     ((dx > 0) - (dx < 0), (dy > 0) - (dy < 0)))
        plane = DIRECTIONS.index(direction) + (10 if move.promotion else 0)
    return plane * 81 + destination


def legal_actions(board):
    result = {}
    for move in board.legal_moves:
        index = action_index(board, move)
        if index in result:
            raise ValueError('Policy representation collision in legal moves')
        result[index] = move
    return result


def feature_key(board):
    """Canonical board/hands + file-mirror; group repeated states together."""
    x = encode(board)[:42]
    return min(hashlib.sha256(x.tobytes()).hexdigest(),
               hashlib.sha256(x[:, :, ::-1].copy().tobytes()).hexdigest())
