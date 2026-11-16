from collections import Counter
from contextlib import nullcontext
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import shogi

from shogi_auto.engine import SearchStopped
from shogi_auto.policy.features import legal_actions
from shogi_auto.policy.search import choose_policy_move


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def uniform(board, value=0.0):
    moves = [move for move in board.legal_moves]
    return {move: 1 / len(moves) for move in moves}, value


class PolicyBudgetTests(unittest.TestCase):
    def test_zero_budget_and_preset_stop_return_legal_without_evaluation(self):
        board = shogi.Board()
        stop = threading.Event(); stop.set()
        for kwargs in ({'seconds': 0}, {'stop': stop}):
            with self.subTest(kwargs=kwargs):
                def forbidden(_):
                    self.fail('Expired search must not call the evaluator')
                result = choose_policy_move(board, forbidden, **kwargs)
                self.assertTrue(board.is_legal(result.move))
                self.assertEqual(result.nodes, 0)

    def test_terminal_board_does_not_call_evaluator(self):
        board = shogi.Board('4k4/4G4/4R4/9/9/9/9/9/4K4 w - 1')
        result = choose_policy_move(board, lambda _: self.fail('Terminal evaluation'))
        self.assertIsNone(result.move)
        self.assertEqual(result.score, -100000)

    def test_cooperative_root_timeout_is_a_legal_fallback(self):
        clock = Clock()
        class SlowEvaluator:
            calls = 0
            def __call__(self, board):
                raise AssertionError('Use the cooperative method')
            def evaluate_with_budget(self, board, check_budget):
                for _ in range(100):
                    clock.now += .02
                    self.calls += 1
                    check_budget()
                return uniform(board)
        evaluator = SlowEvaluator()
        board = shogi.Board()
        with patch('shogi_auto.policy.search.time.monotonic', clock):
            result = choose_policy_move(board, evaluator, seconds=.05)
        self.assertTrue(board.is_legal(result.move))
        self.assertEqual(evaluator.calls, 3)
        self.assertEqual(result.nodes, 0)
        self.assertAlmostEqual(clock.now, .06)

    def test_noncooperative_call_cannot_be_preempted_or_counted_as_completed(self):
        clock = Clock()
        def slow(board):
            clock.now += .2
            return uniform(board)
        board = shogi.Board()
        with patch('shogi_auto.policy.search.time.monotonic', clock):
            result = choose_policy_move(board, slow, seconds=.05)
        self.assertTrue(board.is_legal(result.move))
        self.assertEqual(result.nodes, 0)
        self.assertEqual(clock.now, .2)

    def test_network_checkpoint_discards_an_overdue_forward_result(self):
        # Exercise the production Evaluator's control flow with a fake forward,
        # without claiming tensor correctness or requiring PyTorch for CPU CI.
        from shogi_auto.policy.network import Evaluator
        clock = Clock()
        calls = []
        def forward(_):
            calls.append(1)
            clock.now += .2
            return None, None
        evaluator = object.__new__(Evaluator)
        evaluator.network = forward
        fake_torch = SimpleNamespace(no_grad=nullcontext, from_numpy=lambda _: {None: object()})
        board = shogi.Board()
        with patch.dict(sys.modules, {'torch': fake_torch}), \
                patch('shogi_auto.policy.network.encode', return_value=object()), \
                patch('shogi_auto.policy.search.time.monotonic', clock):
            result = choose_policy_move(board, evaluator, seconds=.05)
        self.assertEqual(calls, [1])
        self.assertTrue(board.is_legal(result.move))
        self.assertEqual(result.nodes, 0)
        self.assertEqual(clock.now, .2)

    def test_known_simulation_cost_leaves_reserve(self):
        clock = Clock()
        calls = []
        def evaluate(board):
            calls.append(board.sfen())
            clock.now += .04
            return uniform(board)
        with patch('shogi_auto.policy.search.time.monotonic', clock):
            result = choose_policy_move(shogi.Board(), evaluate, seconds=.1)
        self.assertEqual(result.nodes, 2)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(clock.now, .08)

    def test_cancelled_leaf_retains_completed_choice_and_caller_history(self):
        board = shogi.Board()
        for move in ('7g7f', '3c3d'):
            board.push_usi(move)
        before = board.sfen(), list(board.move_stack), Counter(board.transpositions)
        target = list(board.legal_moves)[1]
        stop = threading.Event()
        calls = []
        def evaluate(position):
            calls.append(position.sfen())
            priors, _ = uniform(position)
            if len(calls) == 1:
                priors = {move: .99 if move == target else .0001 for move in priors}
            if len(calls) == 3:
                stop.set()
            return priors, -.7
        result = choose_policy_move(board, evaluate, seconds=10, stop=stop)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.nodes, 2)
        self.assertEqual(result.move, target)
        self.assertEqual(result.score, 700)
        self.assertEqual(board.sfen(), before[0])
        self.assertEqual(list(board.move_stack), before[1])
        self.assertEqual(+board.transpositions, +before[2])

    def test_stop_during_descent_skips_next_evaluation(self):
        board = shogi.Board()
        stop = threading.Event()
        original = shogi.Board.is_fourfold_repetition
        calls = []
        def repetition(position):
            if len(position.move_stack) >= 2:
                stop.set()
            return original(position)
        def evaluate(position):
            calls.append(position.sfen())
            priors, value = uniform(position)
            # Follow the same child on the third simulation.
            first = next(iter(priors))
            return {move: 1.0 if move == first else 0.0 for move in priors}, value
        with patch.object(shogi.Board, 'is_fourfold_repetition', repetition):
            result = choose_policy_move(board, evaluate, seconds=10, stop=stop)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.nodes, 2)
        self.assertEqual(result.depth, 1)
        self.assertEqual(board.sfen(), shogi.Board().sfen())

    def test_unrelated_evaluator_errors_are_not_swallowed(self):
        def broken(_):
            raise ValueError('bad model')
        with self.assertRaisesRegex(ValueError, 'bad model'):
            choose_policy_move(shogi.Board(), broken)

    def test_legal_labels_can_stop_between_yields_without_mutating_board(self):
        board = shogi.Board()
        before = board.sfen(), list(board.move_stack)
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 4:
                raise SearchStopped
        with self.assertRaises(SearchStopped):
            legal_actions(board, check_budget=check)
        self.assertEqual(len(calls), 4)
        self.assertEqual((board.sfen(), list(board.move_stack)), before)

    def test_invalid_budgets_fail_before_evaluation(self):
        for seconds in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                choose_policy_move(shogi.Board(), uniform, seconds=seconds)
        for simulations in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                choose_policy_move(shogi.Board(), uniform, max_simulations=simulations)


if __name__ == '__main__':
    unittest.main()
