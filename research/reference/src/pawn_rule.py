"""Isolated python-shogi rule correction; never patches installed Board.
Direct checking pawn drop cannot be blocked and a reply drop cannot capture.
Thus only board moves that leave the defender king safe can evade it.
No C result, NN, mate search, repetition or terminal scoring is used.
"""
import shogi
class RuleBoard(shogi.Board):
 def was_check_by_dropping_pawn(self,move):
  if move.drop_piece_type!=shogi.PAWN:return False
  king=self.king_squares[self.turn]
  if king is None:return False
  attacks=shogi.BB_PAWN_ATTACKS[self.turn^1][move.to_square]
  if not attacks & shogi.BB_SQUARES[king]:return False
  for reply in self.generate_pseudo_legal_moves():
   if reply.from_square is None:continue
   self.push(reply)
   try:safe=not self.was_suicide()
   finally:self.pop()
   if safe:return False
  return True
