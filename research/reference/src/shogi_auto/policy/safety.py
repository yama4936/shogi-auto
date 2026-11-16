"""Optional selected-move one-ply mate guard within the same total clock."""
from ..board import new_board,clone_board,require_rule_board,push_legal_usi,is_rule_board
import copy
from dataclasses import dataclass
import math
import time
from ..engine import SearchResult,SearchStopped
from .tactics import choose_tactical_move,find_checking_mate

@dataclass(frozen=True)
class GuardedResult:
    search: SearchResult
    status: str
    rejected: tuple
    elapsed_seconds: float


def choose_guarded_move(board,evaluator,*,seconds=.1,max_simulations=4096,mate_fraction=.2,
                        safety_fraction=.2,max_alternatives=8,stop=None):
    require_rule_board(board)
    if not math.isfinite(safety_fraction) or not 0<=safety_fraction<=.5:raise ValueError('Safety fraction must be in [0,.5]')
    if not isinstance(max_alternatives,int) or max_alternatives<1:raise ValueError('Positive alternative limit required')
    started=time.monotonic();deadline=started+seconds
    base=choose_tactical_move(board,evaluator,seconds=seconds*(1-safety_fraction),max_simulations=max_simulations,
        mate_fraction=mate_fraction,mate_plies=5,mate_nodes=4000,stop=stop)
    original=base.search;rejected=[]
    def result(search,status):return GuardedResult(search,status,tuple(rejected),time.monotonic()-started)
    if original.move is None or base.mate.move is not None or safety_fraction==0:return result(original,'not_needed')
    position=clone_board(board)
    def check():
        if time.monotonic()>=deadline or (stop is not None and stop.is_set()):raise SearchStopped
    def inspect(move):
        check();position.push(move)
        try:
            proof=find_checking_mate(position,seconds=min(.02,max(0.,deadline-time.monotonic())),
                max_plies=1,max_nodes=10000,stop=stop)
            check()
            return proof
        finally:position.pop()
    try:
        proof=inspect(original.move)
        if proof.move is None:
            return result(original,'no_mate_in_one' if proof.status=='no_checking_mate_within_bound' else 'unknown')
        rejected.append(original.move.usi());check()
        cooperative=getattr(evaluator,'evaluate_with_budget',None)
        priors,_=cooperative(position,check) if cooperative is not None else evaluator(position)
        check()
        alternatives=[m for m in sorted(priors,key=priors.get,reverse=True) if m!=original.move and position.is_legal(m)]
        for move in alternatives[:max_alternatives]:
            proof=inspect(move)
            if proof.move is not None:rejected.append(move.usi())
            elif proof.status=='no_checking_mate_within_bound':
                # This certifies only absence of an immediate checking mate,
                # not positional safety or a win. Keep NN simulation count.
                selected=SearchResult(move,original.depth,original.nodes,0)
                return result(selected,'replaced_proved_mate_in_one')
        return result(original,'no_verified_alternative')
    except SearchStopped:return result(original,'unknown')
