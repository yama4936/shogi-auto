from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import shogi
from shogi_auto.engine import SearchResult
from shogi_auto.policy.safety import choose_guarded_move
from shogi_auto.policy.tactics import find_checking_mate

class MateSafetyTests(unittest.TestCase):
    def test_proved_blunder_replaced_by_completely_checked_alternative(self):
        b=shogi.Board('k8/9/9/9/9/9/2P6/5r3/4K4 b g 1')
        bad=shogi.Move.from_usi('7g7f');good=shogi.Move.from_usi('5i4h');self.assertTrue(b.is_legal(bad));self.assertTrue(b.is_legal(good))
        before=b.sfen(),list(b.move_stack),Counter(b.transpositions)
        def evaluator(board):return {good:1.,bad:.1},0.
        base=SimpleNamespace(search=SearchResult(bad,1,3,0),mate=SimpleNamespace(move=None))
        with patch('shogi_auto.policy.safety.choose_tactical_move',return_value=base):
            result=choose_guarded_move(b,evaluator,seconds=1)
        self.assertEqual(result.search.move,good);self.assertEqual(result.status,'replaced_proved_mate_in_one');self.assertIn(bad.usi(),result.rejected)
        self.assertEqual((b.sfen(),list(b.move_stack),Counter(b.transpositions)),before)
        b.push(good);self.assertEqual(find_checking_mate(b,seconds=1,max_plies=1).status,'no_checking_mate_within_bound')

    def test_unknown_never_claimed_safe_and_zero_budget_legal_fallback(self):
        b=shogi.Board();result=choose_guarded_move(b,lambda _:self.fail('No time'),seconds=0)
        self.assertTrue(b.is_legal(result.search.move));self.assertIn(result.status,('unknown','not_needed'))
        with self.assertRaises(ValueError):choose_guarded_move(b,None,safety_fraction=1)

if __name__=='__main__':unittest.main()
