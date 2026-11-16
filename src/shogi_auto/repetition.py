"""Shared continuous-check rule and history-safe search adjudication."""
import copy
import shogi


def continuous_check_loser(turns):
    for color in (shogi.BLACK,shogi.WHITE):
        checks=[check for side,check in turns if side==color]
        if checks and all(checks):return color
    return None


def repetition_value(board,check_budget=lambda:None):
    """Side-to-move value at fourfold repetition: checker loses, others draw.

    Reconstruct only the last four occurrences from an independent board copy.
    The caller's push/pop stacks and transposition counters stay untouched.
    """
    if not board.is_fourfold_repetition():raise ValueError('Repetition required')
    check_budget();position=copy.deepcopy(board)
    key=lambda b:' '.join(b.sfen().split()[:3])
    target=key(position);seen=1;turns=[]
    while position.move_stack:
        check_budget()
        turns.append((1-position.turn,position.is_check()))
        position.pop()
        if key(position)==target:
            seen+=1
            if seen==4:
                loser=continuous_check_loser(turns)
                return 0. if loser is None else (-1. if loser==board.turn else 1.)
    raise ValueError('Fourfold repetition has insufficient recoverable history')
