import socket
import threading
import unittest

import shogi

from shogi_auto.csa import CommunicationError,LineTransport,ProtocolError,Session,Summary
from shogi_auto.policy.tournament import CSA_KINDS,OPENING,csa_move


def position_lines():
    board=shogi.Board();lines=[]
    for rank in range(9):
        row=f'P{rank+1}'
        for square in range(rank*9,rank*9+9):
            p=board.piece_at(square)
            row+=('+' if p.color==0 else '-')+CSA_KINDS[p.piece_type] if p else ' * '
        lines.append(row)
    return lines+['P+','P-','+']


def summary(color=0,history=(),increment='2'):
    lines=['BEGIN Game_Summary','Protocol_Version:1.2','Protocol_Mode:Server','Format:Shogi 1.0',
           'Declaration:Jishogi 1.1','Game_ID:dr7-local','Name+:fixture-black','Name-:fixture-white',
           'Your_Turn:'+('+' if color==0 else '-'),'To_Move:'+('+' if len(history)%2==0 else '-'),
           'Max_Moves:512']
    for sign,total in [('+','180'),('-','600')]:
        lines+=['BEGIN Time'+sign,'Time_Unit:1sec','Total_Time:'+total,'Increment:'+increment,
                'Time_Roundup:NO','Least_Time_Per_Move:0','END Time'+sign]
    lines+=['BEGIN Position']+position_lines()+list(history)+['END Position','END Game_Summary']
    return lines


class CSATests(unittest.TestCase):
    def pair(self,server_function):
        client,peer=socket.socketpair();client.settimeout(2);peer.settimeout(2);errors=[]
        def worker():
            try:
                with peer,peer.makefile('r') as reader:server_function(peer,reader)
            except BaseException as exc:errors.append(exc)
        thread=threading.Thread(target=worker);thread.start()
        self.addCleanup(client.close)
        def finish():
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(),'mock server thread did not terminate')
            if errors:raise errors[0]
        self.addCleanup(finish)
        return Session(client,2,max_receive_seconds=2)

    def send_lines(self,peer,lines):peer.sendall(('\n'.join(lines)+'\n').encode())

    def handshake(self,peer,reader,color=0,history=()):
        self.assertEqual(reader.readline(),'LOGIN fixture fixture-password\n')
        # Fragment the first reply and coalesce the remaining summary lines.
        peer.sendall(b'LOGIN:fix');peer.sendall(b'ture OK\n')
        self.send_lines(peer,summary(color,history))
        self.assertEqual(reader.readline(),'AGREE dr7-local\n');peer.sendall(b'START:dr7-local\n')

    def test_official_advertised_version_nested_time_and_restored_history(self):
        def server(peer,reader):
            self.handshake(peer,reader,history=['+5958OU,T0','-5152OU,T0','+5859OU,T1','-5251OU,T0'])
            self.assertEqual(reader.readline(),'%TORYO\n')
            self.send_lines(peer,['%TORYO,T0','#RESIGN','#LOSE'])
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        self.assertEqual(len(session.game.history),4);self.assertEqual(session.clock.remaining,[185,604])
        result=session.run(lambda board,remaining:'%TORYO')
        self.assertEqual(result['outcome'],'loss');self.assertTrue(result['server_confirmed'])
        self.assertNotIn('fixture-password',str(session.transport.trace))

    def test_black_initial_four_echo_once_authoritative_server_time_and_logout(self):
        def server(peer,reader):
            self.handshake(peer,reader)
            self.assertEqual(reader.readline(),'+5958OU\n')
            self.send_lines(peer,['+5958OU,T2','-5152OU,T0'])
            self.assertEqual(reader.readline(),'+5859OU\n')
            self.send_lines(peer,['+5859OU,T0','-5251OU,T0'])
            self.assertEqual(reader.readline(),'%TORYO\n')
            self.send_lines(peer,['%TORYO,T0','#RESIGN','#LOSE'])
            self.assertEqual(reader.readline(),'LOGOUT\n');peer.sendall(b'LOGOUT:completed\n')
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        calls=[]
        result=session.run(lambda board,remaining:(calls.append(remaining) or '%TORYO'))
        self.assertEqual(result['normal_score'],0);self.assertEqual(len(session.game.history),4)
        self.assertEqual(calls,[184]);self.assertEqual(session.clock.remaining,[184,604]);session.logout()

    def test_white_waits_for_opponent_and_does_not_send_out_of_turn(self):
        def server(peer,reader):
            self.handshake(peer,reader,color=1)
            peer.sendall(b'+5958OU,T0\n');self.assertEqual(reader.readline(),'-5152OU\n')
            self.send_lines(peer,['-5152OU,T0','+5859OU,T0'])
            self.assertEqual(reader.readline(),'-5251OU\n')
            self.send_lines(peer,['-5251OU,T0','%TORYO,T0','#RESIGN','#WIN'])
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        result=session.run(lambda *_:self.fail('white player was called out of turn'))
        self.assertEqual(result['outcome'],'win');self.assertEqual(len(session.game.history),4)

    def test_mismatched_profile_never_agrees(self):
        def server(peer,reader):
            reader.readline();peer.sendall(b'LOGIN:fixture OK\n');self.send_lines(peer,summary(increment='3'))
            self.assertEqual(reader.readline(),'')
        session=self.pair(server);self.assertFalse(session.start('fixture','fixture-password'));session.close()
        self.assertEqual(session.result['phase'],'summarizing');self.assertFalse(session.result['server_confirmed'])
        self.assertFalse(any(value.startswith('AGREE') for kind,value in session.transport.trace if kind=='send'))

    def test_login_failure_and_partial_summary_disconnect_are_not_retried(self):
        for assigned in (False,True):
            def server(peer,reader,assigned=assigned):
                reader.readline()
                if assigned:self.send_lines(peer,['LOGIN:fixture OK','BEGIN Game_Summary','Your_Turn:-'])
                else:peer.sendall(b'LOGIN:incorrect\n')
            session=self.pair(server);self.assertFalse(session.start('fixture','fixture-password'))
            self.assertFalse(session.result['automatic_reconnect'])
            self.assertEqual(session.result['normal_score'],0 if assigned else None)

    def test_early_play_disconnect_is_local_loss_without_fabricated_server_result(self):
        def server(peer,reader):
            self.handshake(peer,reader);self.assertEqual(reader.readline(),'+5958OU\n')
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        result=session.run(lambda *_:'%TORYO')
        self.assertEqual(result['outcome'],'local_failure');self.assertEqual(result['normal_score'],0)
        self.assertFalse(result['server_confirmed']);self.assertEqual(len(session.game.history),0)

    def test_server_draw_and_limit_scores_keep_color_asymmetry(self):
        for color in (0,1):
            def server(peer,reader,color=color):
                self.handshake(peer,reader,color=color)
                self.send_lines(peer,['#MAX_MOVES','#CENSORED'])
            session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
            # Boundary state is synthetic here; actual 511/512 move priority is
            # covered separately by the rules tests.
            session.game.history=[{} for _ in range(512)];session.waiting_result=True
            result=session.run(lambda *_:'%TORYO')
            self.assertEqual(result['normal_score'],.5);self.assertEqual(result['tournament_score'],.4 if color==0 else .6)

    def test_special_declaration_result_and_time_up(self):
        from shogi_auto.game import Game
        from test_tournament import declaration_board
        def server(peer,reader):
            self.handshake(peer,reader)
            self.assertEqual(reader.readline(),'%KACHI\n')
            self.send_lines(peer,['%KACHI,T0','#JISHOGI','#WIN'])
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        session.game=Game(sfen=declaration_board().sfen());session.game.history=[{} for _ in range(4)]
        self.assertEqual(session.run(lambda *_:'%KACHI')['outcome'],'win')
        def timeout_server(peer,reader):
            self.handshake(peer,reader)
            self.assertEqual(reader.readline(),'+5958OU\n')
            self.send_lines(peer,['#TIME_UP','#LOSE'])
        session=self.pair(timeout_server);self.assertTrue(session.start('fixture','fixture-password'))
        self.assertEqual(session.run(lambda *_:'%TORYO')['outcome'],'loss')

    def test_invalid_login_frames_and_pending_echo_are_rejected(self):
        c,p=socket.socketpair();self.addCleanup(c.close);self.addCleanup(p.close)
        session=Session(c)
        with self.assertRaises(ProtocolError):session.login('bad user','safe')
        with self.assertRaises(ProtocolError):session.login('fixture','bad\npassword')
        with self.assertRaises(ProtocolError):session.send_decision('7g7f')
        with self.assertRaises(ProtocolError):LineTransport.valid('bad\tline')

    def test_mating_side_waits_for_resignation_and_mated_side_resigns(self):
        from shogi_auto.game import Game
        for color in (0,1):
            def server(peer,reader,color=color):
                self.handshake(peer,reader,color=color)
                if color==0:self.assertEqual(reader.readline(),'+0052KI\n')
                peer.sendall(b'+0052KI,T0\n')
                if color==1:self.assertEqual(reader.readline(),'%TORYO\n')
                self.send_lines(peer,['%TORYO,T0','#RESIGN','#WIN' if color==0 else '#LOSE'])
            session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
            session.game=Game(sfen='4k4/9/4R4/9/9/9/9/9/4K4 b G 1')
            session.game.history=[{} for _ in range(4)]
            result=session.run(lambda *_:'G*5b' if color==0 else self.fail('mated side should resign without calling search'))
            self.assertEqual(result['outcome'],'win' if color==0 else 'loss')
            self.assertTrue(result['server_confirmed'])

    def test_optional_game_id_and_interruption_have_no_fabricated_score(self):
        def server(peer,reader):
            reader.readline();peer.sendall(b'LOGIN:fixture OK\n')
            self.send_lines(peer,[line for line in summary(1) if not line.startswith('Game_ID:')])
            self.assertEqual(reader.readline(),'AGREE\n');peer.sendall(b'START:\n#CHUDAN\n')
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        result=session.run(lambda *_:self.fail('white must wait'))
        self.assertEqual(result['outcome'],'interrupted');self.assertIsNone(result['normal_score'])

    def test_divergent_own_echo_stops_without_applying_a_move(self):
        def server(peer,reader):
            self.handshake(peer,reader);self.assertEqual(reader.readline(),'+5958OU\n')
            peer.sendall(b'+5859OU,T0\n')
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        result=session.run(lambda *_:'%TORYO')
        self.assertEqual(result['outcome'],'local_failure');self.assertEqual(len(session.game.history),0)
        self.assertFalse(result['server_confirmed']);self.assertFalse(result['automatic_reconnect'])

    def test_illegal_opponent_echo_waits_for_server_adjudication(self):
        def server(peer,reader):
            self.handshake(peer,reader,color=1)
            self.send_lines(peer,['+7776HI,T0','#ILLEGAL_MOVE','#WIN'])
        session=self.pair(server);self.assertTrue(session.start('fixture','fixture-password'))
        result=session.run(lambda *_:self.fail('no white move required'))
        self.assertEqual(result['outcome'],'win');self.assertTrue(result['server_confirmed'])
        self.assertEqual(len(session.game.history),0);self.assertIn('invalid_server_move',session.events[-1])


if __name__=='__main__':unittest.main()
