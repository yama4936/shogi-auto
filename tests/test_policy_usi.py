import io,unittest
import shogi
from shogi_auto.policy.usi import go_budget,run,forced_opening

class PolicyUSITests(unittest.TestCase):
    def test_both_fischer_clocks_and_reserve(self):
        self.assertEqual(go_budget('go btime 180000 wtime 600000 binc 2000 winc 2000'.split(),0),(2.,182.))
        self.assertEqual(go_budget('go btime 0 wtime 600000 binc 2000 winc 2000'.split(),0),(.04375,2.))
        self.assertEqual(go_budget('go movetime 100'.split(),0),(0.,.1))
        for text in ('go','go btime 1','go movetime -1','go movetime 1 movetime 2','go ponder','go btime 1 wtime 1 byoyomi 1000'):
            with self.assertRaises(ValueError):go_budget(text.split(),0)

    def test_explicit_model_stop_and_dr7_initial_reply(self):
        def uniform(board):
            moves=list(board.legal_moves);return {m:1/len(moves) for m in moves},0.
        output=io.StringIO();run(io.StringIO('usi\nisready\nposition startpos\ngo btime 180000 wtime 600000 binc 2000 winc 2000\nstop\nquit\n'),output,uniform,dr7_opening=True)
        text=output.getvalue();self.assertIn('usiok',text);self.assertIn('readyok',text);self.assertEqual(text.count('bestmove '),1);self.assertIn('bestmove 5i5h',text)

    def test_sfen_initial_layout_at_move_five_is_not_first_turn(self):
        b=shogi.Board()
        for move in ('5i5h','5a5b','5h5i','5b5a'):b.push_usi(move)
        restored=shogi.Board(b.sfen())
        self.assertEqual(restored.sfen().split()[-1],'5')
        self.assertEqual(len(restored.move_stack),0)
        self.assertIsNone(forced_opening(restored))

    def test_initial_four_and_custom_sfen(self):
        b=shogi.Board()
        for move in ('5i5h','5a5b','5h5i','5b5a'):
            self.assertEqual(forced_opening(b),move);b.push_usi(move)
        self.assertIsNone(forced_opening(b))
        self.assertIsNone(forced_opening(shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b - 1')))

if __name__=='__main__':unittest.main()
