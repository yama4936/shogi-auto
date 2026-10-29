from collections import Counter
import unittest

import shogi

from shogi_auto.engine import MATE, choose_move, evaluate, ordered_moves
from shogi_auto.game import Game


def exhaustive(board, depth, ply=0):
    if board.is_fourfold_repetition():
        return 0
    legal = [move for move in board.legal_moves]
    if not legal:
        return -MATE + ply
    if depth == 0:
        return evaluate(board)
    scores = []
    for move in legal:
        board.push(move)
        try:
            scores.append(-exhaustive(board, depth - 1, ply + 1))
        finally:
            board.pop()
    return max(scores)


class SearchCorrectnessTests(unittest.TestCase):
    def test_fixed_depth_values_match_exhaustive_search(self):
        for sfen in ('4k4/9/9/4r4/4R4/9/9/9/4K4 b - 1',
                     '4k4/9/9/9/9/9/9/9/4K4 b P 1'):
            for depth in (1, 2):
                board = shogi.Board(sfen)
                expected = exhaustive(board, depth)
                result = choose_move(board, seconds=2, max_depth=depth, max_nodes=100000)
                self.assertEqual(result.depth, depth)
                self.assertEqual(result.score, expected)
                self.assertIn(result.move, board.legal_moves)

    def test_sort_does_not_request_expensive_legal_generator_length(self):
        board = shogi.Board()
        generator_type = type(board.legal_moves)
        from unittest.mock import patch
        with patch.object(generator_type, '__len__', side_effect=AssertionError('length hint enumerates moves')):
            self.assertEqual(len(ordered_moves(board)), 30)

    def test_material_and_hand_signs(self):
        for hand, expected in [('R', 1000), ('2Pp', 100), ('B', 850), ('r', -1000)]:
            board = shogi.Board(f'4k4/9/9/9/9/9/9/9/4K4 b {hand} 1')
            self.assertEqual(evaluate(board), expected)
            board.turn = shogi.WHITE
            self.assertEqual(evaluate(board), -expected)

    def test_partial_first_iteration_keeps_evaluated_choice(self):
        from unittest.mock import patch
        board = shogi.Board()
        ordered = ordered_moves(board)
        def evaluator(position):
            return -100 if position.move_stack and position.move_stack[-1] == ordered[1] else 0
        # Deadline, first root+leaf, second root+leaf, then stop before third.
        with patch('shogi_auto.engine.time.monotonic', side_effect=[0, 0, 0, 0, 0, 1]):
            result = choose_move(board, seconds=.1, evaluator=evaluator)
        self.assertEqual(result.depth, 0)
        self.assertEqual(result.root_moves_evaluated, 2)
        self.assertEqual(result.root_legal_moves, 30)
        self.assertEqual(result.move, ordered[1])
        self.assertEqual(result.score, 100)
        self.assertEqual(board.sfen(), shogi.Board().sfen())

    def test_repetition_count_and_caller_state_survive_search(self):
        game = Game(sfen='4k4/9/9/9/9/9/9/9/4K4 b - 1')
        for _ in range(2):
            for move in ('5i6i', '5a6a', '6i5i', '6a5a'):
                game.push(move)
        before = (game.board.sfen(), list(game.board.move_stack), Counter(game.board.transpositions))
        result = choose_move(game.board, seconds=0.03)
        self.assertIn(result.move, game.board.legal_moves)
        # Legal generation can add zero counters; positive history is invariant.
        self.assertEqual(game.board.sfen(), before[0])
        self.assertEqual(list(game.board.move_stack), before[1])
        self.assertEqual(+game.board.transpositions, +before[2])
        for move in ('5i6i', '5a6a', '6i5i', '6a5a'):
            game.push(move)
        self.assertEqual(game.result['reason'], '千日手')


if __name__ == '__main__':
    unittest.main()
