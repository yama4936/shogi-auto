import threading
import unittest

import shogi

from shogi_auto.engine import choose_move, evaluate
from shogi_auto.game import Game
from shogi_auto.usi import parse_position, run
from io import StringIO


class GameTests(unittest.TestCase):
    def test_opening_and_ai_responses(self):
        game = Game()
        game.human_move('7g7f')
        game.ai_move(seconds=0.03)
        self.assertEqual(len(game.history), 2)
        self.assertEqual(game.board.turn, shogi.BLACK)
        for _ in range(2):
            move = next(iter(game.board.legal_moves))
            game.human_move(move.usi())
            game.ai_move(seconds=0.03)
        self.assertEqual(len(game.history), 6)
        self.assertTrue(game.snapshot()['legal'])

    def test_illegal_and_malformed_moves_do_not_change_position(self):
        game = Game()
        before = game.board.sfen()
        for move in ('7g7e', '5i5a', 'P*5e', '0000', 'oops', '', None, [], '9z1a'):
            with self.assertRaises(ValueError):
                game.human_move(move)
            self.assertEqual(game.board.sfen(), before)
        game.human_move('7g7f')
        with self.assertRaisesRegex(ValueError, 'AI'):
            game.human_move('2g2f')

    def test_capture_and_drop(self):
        game = Game()
        for move in ('7g7f', '3c3d', '8h2b+', '3a2b', 'B*4e'):
            game.push(move)
        self.assertEqual(game.board.piece_at(shogi.SQUARE_NAMES.index('4e')).piece_type, shogi.BISHOP)
        self.assertEqual(game.board.pieces_in_hand[shogi.BLACK].get(shogi.BISHOP, 0), 0)
        self.assertEqual(game.board.pieces_in_hand[shogi.WHITE][shogi.BISHOP], 1)
        self.assertIn('打', game.history[-1]['label'])

    def test_optional_and_forced_promotion(self):
        game = Game(sfen='4k4/P8/9/9/9/9/9/9/4K4 b - 1')
        self.assertIn('9b9a+', game.snapshot()['legal'])
        self.assertNotIn('9b9a', game.snapshot()['legal'])
        with self.assertRaises(ValueError):
            game.push('9b9a')
        game.push('9b9a+')
        self.assertEqual(game.board.piece_at(0).piece_type, shogi.PROM_PAWN)
        game = Game()
        game.push('7g7f'); game.push('3c3d')
        self.assertIn('8h2b', game.snapshot()['legal'])
        self.assertIn('8h2b+', game.snapshot()['legal'])

    def test_drop_restrictions_and_check_evasion(self):
        game = Game(sfen='4k4/9/9/9/9/9/P8/9/4K4 b PLN 1')
        for move in ('P*9e', 'P*8a', 'L*8a', 'N*8a', 'N*8b'):
            with self.assertRaises(ValueError, msg=move):
                game.push(move)
        game.push('P*8e')
        checked = Game(sfen='k3r4/9/9/9/9/9/P8/9/4K4 b - 1')
        self.assertTrue(checked.board.is_check())
        with self.assertRaises(ValueError):
            checked.push('9g9f')

    def test_pawn_drop_mate_is_illegal(self):
        game = Game(sfen='3lkl3/9/4G4/9/9/9/9/9/4K4 b P 1')
        with self.assertRaises(ValueError):
            game.push('P*5b')

    def test_checkmate_and_end_reject_moves(self):
        game = Game()
        for move in ('7g7f', '3c3d', '8h2b+', '4a5b', 'B*4b', '5a4a', '2b3a'):
            game.push(move)
        self.assertEqual(game.result['reason'], '詰み')
        self.assertEqual(game.result['winner'], shogi.BLACK)
        self.assertEqual(game.snapshot()['legal'], [])
        with self.assertRaises(ValueError):
            game.push('4a5a')

    def test_resign_and_new_game(self):
        game = Game(shogi.WHITE)
        game.ai_move(seconds=0.01)
        self.assertEqual(game.board.turn, shogi.WHITE)
        game.resign()
        self.assertEqual(game.result['winner'], shogi.BLACK)
        self.assertEqual(Game().snapshot()['history'], [])

    def test_fourfold_repetition(self):
        game = Game(sfen='4k4/9/9/9/9/9/9/9/4K4 b - 1')
        for _ in range(3):
            for move in ('5i6i', '5a6a', '6i5i', '6a5a'):
                game.push(move)
        self.assertEqual(game.result['reason'], '千日手')
        self.assertIsNone(game.result['winner'])

    def test_perpetual_checker_loses(self):
        game = Game(sfen='4k4/5R3/9/9/9/9/9/9/K8 b - 1')
        for _ in range(3):
            for move in ('4b5b', '5a4a', '5b4b', '4a5a'):
                game.push(move)
        self.assertEqual(game.result['reason'], '連続王手の千日手')
        self.assertEqual(game.result['winner'], shogi.WHITE)


class EngineTests(unittest.TestCase):
    def test_material_capture_and_board_restored(self):
        board = shogi.Board('4k4/9/9/4r4/4R4/9/9/9/4K4 b - 1')
        before = board.sfen(), list(board.move_stack), dict(board.transpositions)
        result = choose_move(board, seconds=1, max_depth=1)
        self.assertEqual(result.move.usi(), '5e5d')
        self.assertEqual((board.sfen(), list(board.move_stack), dict(board.transpositions)), before)
        self.assertEqual(evaluate(shogi.Board()), 0)

    def test_timeout_stop_and_terminal(self):
        board = shogi.Board()
        before = board.sfen()
        stop = threading.Event(); stop.set()
        result = choose_move(board, seconds=0.001, stop=stop)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(board.sfen(), before)
        board = parse_position('startpos moves 7g7f 3c3d 8h2b+ 4a5b B*4b 5a4a 2b3a'.split())
        self.assertIsNone(choose_move(board).move)


class UsiTests(unittest.TestCase):
    def test_handshake_search_and_position(self):
        output = StringIO()
        run(StringIO('usi\nisready\nposition startpos moves 7g7f\ngo movetime 20\nstop\nquit\n'), output)
        lines = output.getvalue().splitlines()
        self.assertIn('usiok', lines)
        self.assertIn('readyok', lines)
        best = next(line.split()[1] for line in lines if line.startswith('bestmove '))
        board = parse_position(['startpos', 'moves', '7g7f'])
        self.assertIn(shogi.Move.from_usi(best), board.legal_moves)
        sfen = board.sfen().split()
        self.assertEqual(parse_position(['sfen', *sfen]).sfen(), board.sfen())

    def test_invalid_position_is_not_pushed(self):
        with self.assertRaises(ValueError):
            parse_position('startpos moves 7g7e'.split())


if __name__ == '__main__':
    unittest.main()
