from collections import Counter
import copy
import threading
import unittest

import shogi

from shogi_auto.policy.tactics import choose_tactical_move, find_checking_mate


def verify_certificate(board, certificate, attacker):
    if certificate.get('mate'):
        assert board.turn != attacker and board.is_checkmate()
    elif 'attack' in certificate:
        assert board.turn == attacker
        move = shogi.Move.from_usi(certificate['attack'])
        assert board.is_legal(move)
        board.push(move)
        assert board.is_check()
        verify_certificate(board, certificate['continuation'], attacker)
        board.pop()
    else:
        assert board.turn != attacker
        legal = {move.usi() for move in board.legal_moves}
        assert legal == set(certificate['defenses'])
        for usi, child in certificate['defenses'].items():
            board.push_usi(usi)
            verify_certificate(board, child, attacker)
            board.pop()


class TacticsTests(unittest.TestCase):
    def test_every_defense_must_be_proved_not_just_first_reply(self):
        class Tree:
            def __init__(self, second_escapes):
                self.state = 'root'; self.stack = []
                self.graph = {
                    'root': (0, False, {'5i5h': 'defense'}),
                    'defense': (1, True, {'5a5b': 'first', '5a4a': 'second'}),
                    'first': (0, False, {'5h5g': 'mate'}),
                    'second': (0, False, {'5h4h': 'escape' if second_escapes else 'mate'}),
                    'mate': (1, True, {}),
                    'escape': (1, True, {'4a3a': 'quiet'}),
                    'quiet': (0, False, {}),
                }
            @property
            def turn(self): return self.graph[self.state][0]
            @property
            def legal_moves(self):
                return (shogi.Move.from_usi(usi) for usi in self.graph[self.state][2])
            def push(self, move):
                previous = self.state
                self.state = self.graph[self.state][2][move.usi()]
                self.stack.append(previous)
            def pop(self): self.state = self.stack.pop()
            def is_check(self): return self.graph[self.state][1]
            def is_fourfold_repetition(self): return False
        unknown = find_checking_mate(Tree(True), seconds=1, max_plies=3)
        self.assertIsNone(unknown.move)
        proved = find_checking_mate(Tree(False), seconds=1, max_plies=3)
        self.assertEqual(proved.status, 'proved')
        self.assertEqual(proved.plies, 3)
        self.assertEqual(set(proved.certificate['continuation']['defenses']), {'5a5b', '5a4a'})

    def test_mate_is_proved_and_certificate_covers_all_defenses(self):
        board = shogi.Board('4k4/9/4R4/9/9/9/9/9/4K4 b G 1')
        before = board.sfen(), list(board.move_stack), +Counter(board.transpositions)
        result = find_checking_mate(board, seconds=2, max_plies=3, max_nodes=10000)
        self.assertEqual(result.status, 'proved')
        self.assertEqual(result.plies, 1)
        verify_certificate(copy.deepcopy(board), result.certificate, board.turn)
        self.assertEqual((board.sfen(), list(board.move_stack), +Counter(board.transpositions)), before)

    def test_cancel_and_tiny_node_budget_never_fabricate_proof(self):
        board = shogi.Board('4k4/9/4R4/9/9/9/9/9/4K4 b G 1')
        stop = threading.Event(); stop.set()
        for kwargs in ({'seconds': 0}, {'max_nodes': 1}, {'stop': stop}):
            result = find_checking_mate(board, **kwargs)
            self.assertIsNone(result.move)
            self.assertIsNone(result.certificate)
            self.assertIn(result.status, ('cancelled', 'budget_exhausted'))

    def test_completed_one_ply_proof_does_not_scan_unrelated_candidates(self):
        class Tree:
            turn = 0
            stack = []
            @property
            def legal_moves(self):
                if self.turn:
                    return iter(())
                def moves():
                    yield shogi.Move.from_usi('5i5h')
                    raise AssertionError('Unrelated root candidates must not delay a complete proof')
                return moves()
            def push(self, move): self.turn = 1
            def pop(self): self.turn = 0
            def is_check(self): return self.turn == 1
            def is_fourfold_repetition(self): return False
        result = find_checking_mate(Tree(), seconds=1, max_nodes=10)
        self.assertEqual(result.status, 'proved')
        self.assertEqual(result.plies, 1)
        self.assertEqual(result.certificate['continuation'], {'mate': True})

    def test_non_mating_position_does_not_claim_win(self):
        board = shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b - 1')
        result = find_checking_mate(board, seconds=1)
        self.assertIsNone(result.move)
        self.assertEqual(result.status, 'no_checking_mate_within_bound')

    def test_repetition_is_not_a_mate(self):
        board = shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b - 1')
        for _ in range(3):
            for usi in ('5i6i', '5a6a', '6i5i', '6a5a'):
                board.push_usi(usi)
        result = find_checking_mate(board, seconds=1)
        self.assertIsNone(result.move)

    def test_repeated_root_with_available_one_ply_mate_is_not_proved(self):
        board = shogi.Board('4k4/5R3/9/9/9/9/9/9/4K4 b G 1')
        for _ in range(3):
            for usi in ('5i6i', '5a6a', '6i5i', '6a5a'):
                self.assertTrue(board.is_legal(shogi.Move.from_usi(usi)))
                board.push_usi(usi)
        self.assertTrue(board.is_fourfold_repetition())
        result = find_checking_mate(board, seconds=1, max_nodes=10000)
        self.assertIsNone(result.move)
        self.assertIsNone(result.certificate)
        self.assertEqual(result.status, 'repetition')

    def test_proved_mate_skips_evaluator_and_counts_no_policy_simulations(self):
        board = shogi.Board('4k4/9/4R4/9/9/9/9/9/4K4 b G 1')
        def forbidden(_):
            self.fail('Proved mate must not depend on the neural evaluator')
        result = choose_tactical_move(board, forbidden, seconds=2, mate_fraction=.5)
        self.assertEqual(result.mate.status, 'proved')
        self.assertEqual(result.search.nodes, 0)
        self.assertTrue(board.is_legal(result.search.move))

    def test_expired_wrapper_still_returns_legal_fallback(self):
        result = choose_tactical_move(shogi.Board(), lambda _: self.fail('No time'), seconds=0)
        self.assertTrue(shogi.Board().is_legal(result.search.move))

    def test_invalid_bounds_rejected(self):
        for kwargs in ({'seconds': -1}, {'max_plies': 4}, {'max_nodes': 0}):
            with self.assertRaises(ValueError):
                find_checking_mate(shogi.Board(), **kwargs)


if __name__ == '__main__':
    unittest.main()
