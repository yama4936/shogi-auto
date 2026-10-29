"""Validated game state shared by the web API and tests."""
from dataclasses import asdict

import shogi

from .engine import choose_move

SIDES = ("先手", "後手")
NAMES = ("", "歩", "香", "桂", "銀", "金", "角", "飛", "玉", "と", "成香", "成桂", "成銀", "馬", "龍")
RANKS = "一二三四五六七八九"


def position_key(board):
    return " ".join(board.sfen().split()[:3])


class Game:
    def __init__(self, human=shogi.BLACK, sfen=None):
        if human not in (shogi.BLACK, shogi.WHITE):
            raise ValueError("先手か後手を選んでください。")
        self.board = shogi.Board(sfen) if sfen else shogi.Board()
        self.human = human
        self.history = []
        self.positions = [position_key(self.board)]
        self.result = None
        self.version = 0
        self.search = None
        self.adjudicate()

    def adjudicate(self):
        if not any(self.board.legal_moves):
            loser = self.board.turn
            reason = "詰み" if self.board.is_check() else "合法手なし"
            self.result = {"winner": 1 - loser, "reason": reason,
                           "text": f"{reason}。{SIDES[1 - loser]}の勝ちです。"}
            return
        key = position_key(self.board)
        repeated = [i for i, value in enumerate(self.positions) if value == key]
        if len(repeated) >= 4:
            cycle = self.history[repeated[-4]:]
            for color in (shogi.BLACK, shogi.WHITE):
                turns = [move for move in cycle if move["color"] == color]
                if turns and all(move["check"] for move in turns):
                    self.result = {"winner": 1 - color, "reason": "連続王手の千日手",
                                   "text": f"連続王手の千日手。{SIDES[color]}の負けです。"}
                    return
            self.result = {"winner": None, "reason": "千日手", "text": "千日手。引き分けです。"}

    def push(self, text):
        if self.result:
            raise ValueError("対局は終了しています。もう一局を選んでください。")
        if not isinstance(text, str) or len(text) > 5:
            raise ValueError("指し手の形式が正しくありません。")
        try:
            move = shogi.Move.from_usi(text)
        except (ValueError, IndexError, TypeError) as exc:
            raise ValueError("指し手の形式が正しくありません。") from exc
        if move not in self.board.legal_moves:
            raise ValueError("その手は指せません。合法手を選んでください。")
        color = self.board.turn
        piece = self.board.piece_at(move.from_square) if move.from_square is not None else None
        kind = piece.piece_type if piece else move.drop_piece_type
        destination = shogi.SQUARE_NAMES[move.to_square]
        label = f"{'▲' if color == shogi.BLACK else '△'}{destination[0]}{RANKS[ord(destination[1]) - ord('a')]}{NAMES[kind]}"
        if move.drop_piece_type:
            label += "打"
        elif move.promotion:
            label += "成"
        else:
            # Include the origin to disambiguate same-type pieces (not full KIF).
            origin = shogi.SQUARE_NAMES[move.from_square]
            label += f"（{origin[0]}{RANKS[ord(origin[1]) - ord('a')]}）"
        self.board.push(move)
        self.history.append({"usi": text, "label": label, "color": color, "check": self.board.is_check()})
        self.positions.append(position_key(self.board))
        self.version += 1
        self.adjudicate()

    def human_move(self, text):
        if self.board.turn != self.human:
            raise ValueError("AIの手番です。")
        self.push(text)

    def ai_move(self, seconds=0.35):
        if self.result or self.board.turn == self.human:
            return
        result = choose_move(self.board, seconds=seconds)
        self.search = asdict(result)
        self.search["move"] = result.move.usi() if result.move else None
        if result.move:
            self.push(result.move.usi())
        else:
            self.adjudicate()

    def resign(self):
        if self.result:
            raise ValueError("対局は終了しています。")
        self.result = {"winner": 1 - self.human, "reason": "投了",
                       "text": f"あなたの投了。{SIDES[1 - self.human]}の勝ちです。"}
        self.version += 1

    def snapshot(self):
        squares = []
        for square in shogi.SQUARES:
            piece = self.board.piece_at(square)
            squares.append({"square": shogi.SQUARE_NAMES[square],
                            "piece": NAMES[piece.piece_type] if piece else None,
                            "color": piece.color if piece else None,
                            "promoted": bool(piece and piece.piece_type >= shogi.PROM_PAWN)})
        hands = [[{"kind": shogi.PIECE_SYMBOLS[kind].upper(), "name": NAMES[kind], "count": count}
                  for kind in (shogi.ROOK, shogi.BISHOP, shogi.GOLD, shogi.SILVER, shogi.KNIGHT, shogi.LANCE, shogi.PAWN)
                  if (count := self.board.pieces_in_hand[color].get(kind, 0))]
                 for color in (shogi.BLACK, shogi.WHITE)]
        return {"board": squares, "hands": hands, "turn": self.board.turn, "human": self.human,
                "check": self.board.is_check(), "history": self.history, "result": self.result,
                "version": self.version, "sfen": self.board.sfen(), "search": self.search,
                "legal": [move.usi() for move in self.board.legal_moves] if not self.result else []}
