"""A deliberately small piece-square residual model with CPU-only inference.

    Features are signed, color-normalized piece-square indicators and hand counts.
    The learned correction is bounded to +/-600 centipawns. This is not dlshogi
    or NNUE, and does not reuse any third-party model parameters.
"""
import json
import math
from pathlib import Path

import shogi

from ..engine import evaluate

FEATURES = 14 * 81 + 7
SCALE = 1000.0
CORRECTION_CAP = 600.0


def sparse_features(board):
    for square in shogi.SQUARES:
        piece = board.piece_at(square)
        if piece:
            normalized = square if piece.color == shogi.BLACK else 80 - square
            sign = 1.0 if piece.color == board.turn else -1.0
            yield (piece.piece_type - 1) * 81 + normalized, sign
    for color in (shogi.BLACK, shogi.WHITE):
        sign = 1.0 if color == board.turn else -1.0
        for kind, count in board.pieces_in_hand[color].items():
            if count:
                yield 14 * 81 + kind - 1, sign * count


def encode(board):
    vector = [0.0] * FEATURES
    for index, value in sparse_features(board):
        vector[index] += value
    return vector


def position_key(board):
    return ' '.join(board.sfen().split()[:3])


class LinearEvaluator:
    def __init__(self, weights):
        if len(weights) != FEATURES or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in weights):
            raise ValueError('Invalid piece-square model weights')
        self.weights = tuple(weights)

    @classmethod
    def load(cls, path):
        model = json.loads(Path(path).read_text())
        if model.get('schema') != 1 or model.get('features') != FEATURES or model.get('scale') != SCALE:
            raise ValueError('Unsupported model schema')
        return cls(model['weights'])

    def __call__(self, board):
        residual = sum(self.weights[index] * value for index, value in sparse_features(board)) * SCALE
        return evaluate(board) + max(-CORRECTION_CAP, min(CORRECTION_CAP, residual))


def export_model(path, weights, metadata):
    from .common import write_json
    weights = [float(value) for value in weights]
    LinearEvaluator(weights)
    write_json(path, {'schema': 1, 'features': FEATURES, 'scale': SCALE,
                      'correction_cap_cp': CORRECTION_CAP, 'weights': weights, 'metadata': metadata})
