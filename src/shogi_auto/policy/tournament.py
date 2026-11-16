"""Local, auditable DR7 rule logic. Not a live tournament connection."""
from dataclasses import dataclass, field
import math
import re

import shogi

from ..game import Game

RULES_URL = 'https://denryu-sen.jp/denryusen/dr7_prd/dr7_rule.pdf'
RULES_SHA256 = '707c43e5b749a8319240a6b986cc8616f8662424bf47bd9f486e6ec719a3014b'
OPENING = ('5i5h', '5a5b', '5h5i', '5b5a')
CSA_KINDS = ('', 'FU', 'KY', 'KE', 'GI', 'KI', 'KA', 'HI', 'OU',
             'TO', 'NY', 'NK', 'NG', 'UM', 'RY')


@dataclass
class Clock:
    remaining: list = field(default_factory=lambda: [180, 600])
    increment: int = 2
    active_color: int | None = None

    def begin(self,color):
        if self.active_color is None:
            self.remaining[color] += self.increment
            self.active_color=color
        elif self.active_color != color:
            raise ValueError('Previous turn was not acknowledged')

    def elapse(self, color, total_seconds):
        if not math.isfinite(total_seconds) or total_seconds < 0:
            raise ValueError('Elapsed time must include communication and be finite/nonnegative')
        self.begin(color)
        self.remaining[color] -= math.floor(total_seconds)
        return self.remaining[color] > 0

    def moved(self, color):
        if self.active_color != color:raise ValueError('Wrong acknowledged side')
        self.active_color=None


def declaration(board, remaining):
    color = board.turn
    king = board.king_squares[color]
    zone = lambda square: square // 9 < 3 if color == shogi.BLACK else square // 9 >= 6
    if remaining <= 0 or king is None or not zone(king) or board.is_check():
        return False
    count, points = 0, 0
    for square in shogi.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and piece.piece_type != shogi.KING and zone(square):
            count += 1
            points += 5 if piece.piece_type in (shogi.ROOK, shogi.BISHOP, shogi.PROM_ROOK, shogi.PROM_BISHOP) else 1
    for kind, number in board.pieces_in_hand[color].items():
        points += number * (5 if kind in (shogi.ROOK, shogi.BISHOP) else 1)
    return count >= 10 and points >= (28 if color == shogi.BLACK else 27)


class TournamentGame:
    def __init__(self, *, sfen=None):
        self.game = Game(sfen=sfen)
        self.clock = Clock()
        self.result = None

    def finish(self, winner, reason):
        self.result = {'winner': winner, 'reason': reason,
                       'normal_scores': [.5, .5] if winner is None else [int(winner == 0), int(winner == 1)],
                       'tournament_scores': [.4, .6] if winner is None else [int(winner == 0), int(winner == 1)]}
        return self.result

    def play(self, command, total_seconds):
        if self.result:
            raise ValueError('Game already finished')
        color = self.game.board.turn
        if not self.clock.elapse(color, total_seconds):
            return self.finish(1-color, 'time_forfeit')
        if command == '%KACHI':
            legal = declaration(self.game.board, self.clock.remaining[color])
            return self.finish(color if legal else 1-color, 'entering_king' if legal else 'false_declaration')
        if command == '%TORYO':
            return self.finish(1-color, 'resign')
        ply = len(self.game.history)
        if ply < 4 and command != OPENING[ply]:
            return self.finish(1-color, 'opening_violation')
        try:
            self.game.push(command)
        except ValueError:
            return self.finish(1-color, 'illegal_move')
        self.clock.moved(color)
        # The published DR7 limit overrides mate delivered by the boundary move.
        repetition = self.game.result and self.game.result['reason'] in ('千日手','連続王手の千日手')
        if len(self.game.history) >= 512 and not repetition:
            return self.finish(None, '512_move_limit')
        if self.game.result:
            return self.finish(self.game.result['winner'], self.game.result['reason'])
        return None

    def disconnect(self, fault_color=None):
        if self.result is None:
            color=self.game.board.turn if fault_color is None else fault_color
            if color not in (0,1):raise ValueError('Unknown disconnected side')
            return self.finish(1-color, 'communication_failure')
        return self.result


def csa_move(board, usi):
    move = shogi.Move.from_usi(usi)
    if not board.is_legal(move):raise ValueError('Illegal outgoing move')
    origin = '00' if move.from_square is None else shogi.SQUARE_NAMES[move.from_square]
    target = shogi.SQUARE_NAMES[move.to_square]
    square = lambda s: s[0] + str(ord(s[1])-ord('a')+1)
    board.push(move)
    try:kind = CSA_KINDS[board.piece_at(move.to_square).piece_type]
    finally:board.pop()
    return ('+' if board.turn == shogi.BLACK else '-') + (origin if origin == '00' else square(origin)) + square(target) + kind


def parse_csa_move(board, text):
    match = re.fullmatch(r'([+-])(\d\d)([1-9][1-9])([A-Z]{2})(?:,T(\d+))?', text)
    if not match or match[1] != ('+' if board.turn == shogi.BLACK else '-') or match[4] not in CSA_KINDS:
        raise ValueError('Malformed CSA move or wrong side')
    usi_square = lambda s: s[0]+chr(ord('a')+int(s[1])-1)
    kind = CSA_KINDS.index(match[4]);target = usi_square(match[3])
    if match[2] == '00':
        if kind not in range(1, 8):raise ValueError('Invalid drop kind')
        usi = shogi.PIECE_SYMBOLS[kind].upper()+'*'+target
    else:
        if any(c not in '123456789' for c in match[2]):raise ValueError('Invalid origin')
        origin = usi_square(match[2]);piece = board.piece_at(shogi.SQUARE_NAMES.index(origin))
        if piece is None:raise ValueError('No origin piece')
        usi = origin+target+('+' if piece.piece_type != kind else '')
    if csa_move(board, usi) != text.split(',')[0]:raise ValueError('CSA piece kind mismatch')
    return usi, int(match[5]) if match[5] is not None else None


class CSAProtocol:
    """Line-level login/summary/Agree/START state machine; no auto-reconnect."""
    def __init__(self):
        self.state = 'connected'
        self.game_id = None
        self.summary = {}

    def feed(self, line):
        if '\n' in line or '\r' in line:raise ValueError('Pass one decoded line at a time')
        if line.startswith('LOGIN:'):
            if self.state != 'connected' or not line.endswith(' OK'):
                self.state = 'failed';raise ValueError('CSA login failed or duplicated')
            self.state = 'logged_in'
        elif line == 'BEGIN Game_Summary':
            if self.state != 'logged_in':raise ValueError('Summary before login')
            self.state = 'summary'
        elif line == 'END Game_Summary':
            if self.state != 'summary' or 'Game_ID' not in self.summary:raise ValueError('Incomplete summary')
            version = self.summary.get('Protocol_Version', '1.2.1')
            if version not in ('1.2','1.2.1'):raise ValueError('Unsupported CSA protocol')
            self.game_id = self.summary['Game_ID'];self.state = 'ready'
        elif self.state == 'summary' and ':' in line:
            key, value = line.split(':', 1);self.summary[key] = value
        elif line.startswith('START:'):
            if self.state != 'agreed' or line[6:] != self.game_id:raise ValueError('Unexpected START')
            self.state = 'playing'
        elif line.startswith(('REJECT:', '#')):
            self.state = 'finished'

    def agree(self):
        if self.state != 'ready':raise ValueError('Cannot Agree before complete summary or twice')
        self.state = 'agreed'
        return 'AGREE '+self.game_id

    def disconnected(self):
        self.state = 'failed'
        return {'communication_failure': True, 'automatic_reconnect': False}
