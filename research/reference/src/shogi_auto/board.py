"""Single isolated RuleBoard factory. Never patches installed shogi.Board."""
import copy
import shogi
from pawn_rule import RuleBoard

def is_rule_board(board):return type(board) is RuleBoard
def require_rule_board(board):
 if not is_rule_board(board):raise TypeError('isolated RuleBoard required; legacy Board rejected')
 return board
def effective_state(board):
 require_rule_board(board)
 return board.sfen(),tuple(m.usi() for m in board.move_stack),{k:v for k,v in board.transpositions.items() if v}
def new_board(sfen=None):return RuleBoard(sfen) if sfen is not None else RuleBoard()
def clone_board(board):
 before=effective_state(board);cloned=copy.deepcopy(board);require_rule_board(cloned)
 if effective_state(cloned)!=before or effective_state(board)!=before:raise ValueError('clone changed effective history')
 return cloned
def push_legal_usi(board,text):
 require_rule_board(board);move=shogi.Move.from_usi(text)
 if move not in board.legal_moves:raise ValueError('illegal replay move under isolated rules')
 board.push(move)
def restore_board(root_sfen,moves):
 board=new_board(root_sfen)
 for move in moves:push_legal_usi(board,move)
 return board
def serialize_board(board):
 base=clone_board(board);moves=[m.usi() for m in base.move_stack]
 while base.move_stack:base.pop()
 return {'root_sfen':base.sfen(),'moves':moves,'final_sfen':board.sfen()}
