"""Policy/value adapter for Session.run; no remote connection or CLI.

CSA gives its server-rounded clock after increment. Session discounts local
processing; this adapter additionally reserves transport/OS slack. This is a
cooperative budget, not a hard wall-clock or network deadline guarantee.
"""
from ..board import new_board,clone_board,require_rule_board,push_legal_usi,is_rule_board
import math
import time
from .tactics import choose_tactical_move
from .errors import InferenceUnavailable
from .tournament import declaration


def search_seconds(available, *, reserve=.25, horizon=40, cap=2.):
    for value in (available,reserve,horizon,cap):
        if not math.isfinite(value):raise ValueError('Finite clock parameters required')
    if available<0 or reserve<0 or horizon<=0 or cap<0:raise ValueError('Invalid clock parameters')
    return min(cap,max(0.,available-reserve)/horizon)


class ClockedPolicyPlayer:
    def __init__(self,evaluator,*,reserve=.25,horizon=40,cap=2.,max_simulations=4096,mate_fraction=.2,safety_fraction=0.):
        search_seconds(0,reserve=reserve,horizon=horizon,cap=cap)
        if not isinstance(max_simulations,int) or max_simulations<1:raise ValueError('Positive simulation cap required')
        if not math.isfinite(mate_fraction) or not 0<=mate_fraction<=1:raise ValueError('Invalid proof fraction')
        self.evaluator=evaluator;self.reserve=reserve;self.horizon=horizon;self.cap=cap
        self.max_simulations=max_simulations;self.mate_fraction=mate_fraction;self.safety_fraction=safety_fraction;self.records=[]
        if not math.isfinite(safety_fraction) or not 0<=safety_fraction<=.5:raise ValueError("Invalid safety fraction")

    def __call__(self,board,available):
        require_rule_board(board)
        started=time.monotonic()
        try:return self._choose(board,available)
        except (InferenceUnavailable,MemoryError) as exc:
            # Only explicitly classified resource errors are recoverable.
            # Programming, malformed-model and communication faults propagate.
            move=next(iter(board.legal_moves),None)
            command=move.usi() if move else '%TORYO'
            self.records.append({'available_seconds':available,'engine_seconds':time.monotonic()-started,
                'simulations':0,'command':command,'error_class':type(exc).__name__,
                'fallback':'legal_first_move','details_logged':False})
            return command

    def _choose(self,board,available):
        require_rule_board(board)
        seconds=search_seconds(available,reserve=self.reserve,horizon=self.horizon,cap=self.cap)
        started=time.monotonic()
        if declaration(board,available):command='%KACHI';result=None
        elif self.safety_fraction:
            from .safety import choose_guarded_move
            guarded=choose_guarded_move(board,self.evaluator,seconds=seconds,max_simulations=self.max_simulations,mate_fraction=self.mate_fraction,safety_fraction=self.safety_fraction)
            command=guarded.search.move.usi() if guarded.search.move else '%TORYO'
            self.records.append({'available_seconds':available,'search_budget_seconds':seconds,'reserve_seconds':self.reserve,'engine_seconds':time.monotonic()-started,'simulations':guarded.search.nodes,'mate_status':'guarded','safety_status':guarded.status,'rejected':guarded.rejected,'command':command})
            return command
        else:
            result=choose_tactical_move(board,self.evaluator,seconds=seconds,
                max_simulations=self.max_simulations,mate_fraction=self.mate_fraction,mate_plies=5,mate_nodes=4000)
            command=result.search.move.usi() if result.search.move else '%TORYO'
        self.records.append({'available_seconds':available,'search_budget_seconds':seconds,
            'reserve_seconds':self.reserve,'engine_seconds':time.monotonic()-started,
            'simulations':result.search.nodes if result else 0,'mate_status':result.mate.status if result else 'declaration',
            'command':command})
        return command
