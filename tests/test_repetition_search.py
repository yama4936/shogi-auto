from collections import Counter
import unittest
import shogi
from shogi_auto.game import Game
from shogi_auto.repetition import repetition_value
from shogi_auto.policy.search import choose_policy_move

class RepetitionSearchTests(unittest.TestCase):
    def checked_cycle(self,color=0):
        game=Game(sfen='4k4/5R3/9/9/9/9/9/9/K8 b - 1')
        if color:
            board=shogi.Board();board.clear()
            for square in shogi.SQUARES:
                p=game.board.piece_at(square)
                if p:board.set_piece_at(80-square,shogi.Piece(p.piece_type,1-p.color))
            board.turn=1;game=Game(sfen=board.sfen())
        cycle=['4b5b','5a4a','5b4b','4a5a']
        if color:
            cycle=[shogi.Move(80-shogi.Move.from_usi(m).from_square,80-shogi.Move.from_usi(m).to_square).usi() for m in cycle]
        for _ in range(3):
            for move in cycle:game.push(move)
        return game

    def test_both_colors_continuous_check_match_game_without_mutation(self):
        for color in (0,1):
            game=self.checked_cycle(color);b=game.board
            before=b.sfen(),list(b.move_stack),Counter(b.transpositions)
            self.assertEqual(game.result['winner'],1-color)
            expected=-1. if b.turn==color else 1.
            self.assertEqual(repetition_value(b),expected)
            self.assertEqual((b.sfen(),list(b.move_stack),Counter(b.transpositions)),before)
            def forbidden(_):self.fail('Repeated node must not use NN')
            result=choose_policy_move(b,forbidden,seconds=1,max_simulations=2)
            self.assertTrue(b.is_legal(result.move));self.assertEqual(result.nodes,2)

    def test_normal_repetition_is_zero_and_budget_exception_preserves_history(self):
        b=shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b - 1')
        for _ in range(3):
            for move in ('5i6i','5a6a','6i5i','6a5a'):b.push_usi(move)
        self.assertEqual(repetition_value(b),0.)
        before=b.sfen(),list(b.move_stack),Counter(b.transpositions)
        def stop():raise RuntimeError('deadline')
        with self.assertRaisesRegex(RuntimeError,'deadline'):repetition_value(b,stop)
        self.assertEqual((b.sfen(),list(b.move_stack),Counter(b.transpositions)),before)

if __name__=='__main__':unittest.main()
