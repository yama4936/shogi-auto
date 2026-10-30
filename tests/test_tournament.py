import socket
import threading
import unittest

import shogi

from shogi_auto.policy.tournament import (CSAProtocol, Clock, OPENING, TournamentGame,
                                          csa_move, declaration, parse_csa_move)


def declaration_board(color=0, hand_pawns=2):
    board=shogi.Board();board.clear();board.turn=color
    pieces=[('5b',shogi.KING),('1a',shogi.ROOK),('2a',shogi.ROOK),
            ('3a',shogi.BISHOP),('4a',shogi.BISHOP),('6a',shogi.GOLD),
            ('7a',shogi.GOLD),('8a',shogi.GOLD),('9a',shogi.GOLD),
            ('1b',shogi.SILVER),('2b',shogi.SILVER)]
    for square,kind in pieces:
        index=shogi.SQUARE_NAMES.index(square)
        board.set_piece_at(80-index if color else index,shogi.Piece(kind,color))
    enemy=shogi.SQUARE_NAMES.index('9i')
    board.set_piece_at(80-enemy if color else enemy,shogi.Piece(shogi.KING,1-color))
    board.pieces_in_hand[color][shogi.PAWN]=hand_pawns
    return board


class TournamentRulesTests(unittest.TestCase):
    def test_opening_clock_increment_and_color_scores(self):
        game=TournamentGame()
        for move in OPENING:self.assertIsNone(game.play(move,1.999))
        self.assertEqual(game.clock.remaining,[182,602])
        self.assertEqual(len(game.game.history),4)
        self.assertEqual(game.game.board.sfen().split()[:3],shogi.Board().sfen().split()[:3])
        wrong=TournamentGame();self.assertEqual(wrong.play('7g7f',0)['winner'],1)
        result=game.finish(None,'fixture_draw')
        self.assertEqual(result['normal_scores'],[.5,.5]);self.assertEqual(result['tournament_scores'],[.4,.6])

    def test_elapsed_includes_latency_and_floors_fraction(self):
        clock=Clock();self.assertTrue(clock.elapse(0,180.999));self.assertEqual(clock.remaining[0],0)
        self.assertFalse(clock.elapse(0,1.0))
        game=TournamentGame();result=game.play(OPENING[0],181.01)
        self.assertEqual(result['reason'],'time_forfeit');self.assertEqual(result['winner'],1)
        for elapsed in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Clock().elapse(0,elapsed)

    def test_27_point_declarations_and_required_conditions(self):
        self.assertTrue(declaration(declaration_board(0,2),1))
        self.assertFalse(declaration(declaration_board(0,1),1))
        self.assertTrue(declaration(declaration_board(1,1),1))
        self.assertFalse(declaration(declaration_board(1,0),1))
        self.assertFalse(declaration(declaration_board(),0))
        board=declaration_board();square=shogi.SQUARE_NAMES.index('1b');board.remove_piece_at(square)
        board.pieces_in_hand[0][shogi.SILVER]=1
        self.assertFalse(declaration(board,180))
        board=declaration_board();board.set_piece_at(shogi.SQUARE_NAMES.index('5a'),shogi.Piece(shogi.PAWN,1))
        self.assertTrue(board.is_check());self.assertFalse(declaration(board,180))
        board=declaration_board();board.remove_piece_at(shogi.SQUARE_NAMES.index('5b'))
        board.set_piece_at(shogi.SQUARE_NAMES.index('5d'),shogi.Piece(shogi.KING,0))
        self.assertFalse(declaration(board,180))

    def test_declaration_before_increment_false_declaration_loses(self):
        game=TournamentGame(sfen=declaration_board().sfen());game.clock.remaining[0]=0
        self.assertEqual(game.play('%KACHI',0)['reason'],'false_declaration')
        self.assertEqual(game.result['winner'],1)
        game=TournamentGame(sfen=declaration_board().sfen())
        self.assertEqual(game.play('%KACHI',0)['reason'],'entering_king');self.assertEqual(game.result['winner'],0)

    def test_512_boundary_mate_is_draw_and_511_mate_is_win(self):
        for prior,expected in [(510,0),(511,None)]:
            game=TournamentGame(sfen='4k4/9/4R4/9/9/9/9/9/4K4 b G 1')
            game.game.history=[{} for _ in range(prior)]
            result=game.play('G*5b',0)
            self.assertEqual(result['winner'],expected)
            self.assertEqual(result['reason'],'512_move_limit' if expected is None else '詰み')

    def test_csa_roundtrip_opening_promotions_and_drops(self):
        board=shogi.Board()
        expected=['+5958OU','-5152OU','+5859OU','-5251OU']
        for usi,csa in zip(OPENING,expected):
            self.assertEqual(csa_move(board,usi),csa)
            self.assertEqual(parse_csa_move(board,csa+',T2'),(usi,2));board.push_usi(usi)
        board=shogi.Board('4k4/9/9/4P4/9/9/9/9/4K4 b G 1')
        self.assertEqual(parse_csa_move(board,csa_move(board,'5d5c+')),('5d5c+',None))
        self.assertEqual(parse_csa_move(board,csa_move(board,'G*4b')),('G*4b',None))
        with self.assertRaises(ValueError):parse_csa_move(shogi.Board(),'+7776HI')

    def test_socketpair_login_agree_start_and_disconnect_no_retry(self):
        client,server=socket.socketpair();client.settimeout(2);server.settimeout(2)
        seen=[]
        def mock_server():
            with server,server.makefile('r') as reader:
                seen.append(reader.readline().strip())
                server.sendall(b'LOGIN:fixture OK\nBEGIN Game_Summary\nProtocol_Version:1.2.1\nGame_ID:local-fixture\nEND Game_Summary\n')
                seen.append(reader.readline().strip());server.sendall(b'START:local-fixture\n')
        thread=threading.Thread(target=mock_server);thread.start();protocol=CSAProtocol()
        with client,client.makefile('r') as reader:
            client.sendall(b'LOGIN fixture fixture-password\n')
            while protocol.state!='ready':protocol.feed(reader.readline().rstrip('\n'))
            client.sendall((protocol.agree()+'\n').encode())
            with self.assertRaises(ValueError):protocol.agree()
            protocol.feed(reader.readline().rstrip('\n'));self.assertEqual(protocol.state,'playing')
            self.assertEqual(reader.readline(),'');self.assertFalse(protocol.disconnected()['automatic_reconnect'])
        thread.join(timeout=2);self.assertFalse(thread.is_alive())
        self.assertEqual(seen,['LOGIN fixture fixture-password','AGREE local-fixture'])
        for count in range(5):
            game=TournamentGame()
            for move in OPENING[:count]:game.play(move,0)
            self.assertEqual(game.disconnect()['reason'],'communication_failure')
        failed=CSAProtocol()
        with self.assertRaises(ValueError):failed.feed('LOGIN:incorrect')


if __name__=='__main__':unittest.main()
