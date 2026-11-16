import unittest
from unittest.mock import patch
import shogi
from shogi_auto.policy.clock import ClockedPolicyPlayer,search_seconds

class ClockPlayerTests(unittest.TestCase):
    def test_fischer_available_clock_reserve_and_cap(self):
        self.assertEqual(search_seconds(182),2.)
        self.assertEqual(search_seconds(602),2.)
        self.assertAlmostEqual(search_seconds(2.25),.05)
        self.assertEqual(search_seconds(.25),0.)
        self.assertEqual(search_seconds(0),0.)
        for kwargs in ({'available':-1},{'available':float('nan')},{'available':1,'reserve':-1},{'available':1,'horizon':0}):
            with self.assertRaises(ValueError):search_seconds(**kwargs)

    def test_low_clock_returns_legal_without_model_call(self):
        def forbidden(_):self.fail('No inference with exhausted reserve')
        player=ClockedPolicyPlayer(forbidden)
        board=shogi.Board();command=player(board,.1)
        self.assertTrue(board.is_legal(shogi.Move.from_usi(command)))
        self.assertEqual(player.records[-1]['search_budget_seconds'],0.)

    def test_declaration_and_zero_time_boundary(self):
        from test_tournament import declaration_board
        def forbidden(_):self.fail('Declaration must not call model')
        player=ClockedPolicyPlayer(forbidden)
        self.assertEqual(player(declaration_board(),1),'%KACHI')
        command=player(declaration_board(),0)
        self.assertNotEqual(command,'%KACHI')

    def test_only_classified_inference_failure_returns_secret_free_fallback(self):
        from shogi_auto.policy.errors import InferenceUnavailable
        b=shogi.Board()
        def unavailable(_):raise InferenceUnavailable('sensitive backend detail')
        player=ClockedPolicyPlayer(unavailable,cap=1.)
        command=player(b,180)
        self.assertTrue(b.is_legal(shogi.Move.from_usi(command)))
        self.assertEqual(player.records[-1]['error_class'],'InferenceUnavailable')
        self.assertNotIn('sensitive',str(player.records))
        def programming_error(_):raise RuntimeError('programming fault')
        with self.assertRaisesRegex(RuntimeError,'programming fault'):
            ClockedPolicyPlayer(programming_error,cap=1.)(b,180)

    def test_session_discounts_local_delay_without_mutating_server_clock(self):
        from shogi_auto.csa import Session
        from shogi_auto.game import Game
        from shogi_auto.policy.tournament import Clock
        session=Session(None);session.state='playing';session.color=0
        session.game=Game();session.game.history=[{}]*4;session.clock=Clock([2,600]);session.turn_received_at=10.
        observed=[]
        def player(board,available):observed.append(available);return '%TORYO'
        def sent(command):session.pending=command
        def received():session.state='finished';session.result={'fixture':True}
        with patch('shogi_auto.csa.time.monotonic',return_value=11.5),patch.object(session,'send_decision',side_effect=sent),patch.object(session,'receive_event',side_effect=received):
            session.run(player)
        self.assertEqual(observed,[.5]);self.assertEqual(session.clock.remaining,[2,600])
        self.assertEqual(session.events[-1]['send_seconds'],0.)

if __name__=='__main__':unittest.main()
