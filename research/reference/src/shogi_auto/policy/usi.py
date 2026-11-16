"""Explicit-model offline USI adapter with Fischer budgets and stop support.

No remote game connection. Unsupported ponder/infinite/mate modes are rejected.
"""
from ..board import new_board,clone_board,require_rule_board,push_legal_usi,is_rule_board
import argparse
import copy
import math
import sys
import threading
import time
import shogi
from ..usi import parse_position
from .clock import search_seconds
from .tactics import choose_tactical_move
from .tournament import OPENING,declaration


def go_budget(tokens,turn,*,reserve=.25,cap=2.):
    if any(x in tokens for x in ('ponder','infinite','mate')):raise ValueError('Unsupported go mode')
    values={};i=1
    allowed={'movetime','btime','wtime','binc','winc','byoyomi'}
    while i<len(tokens):
        if tokens[i] not in allowed or i+1>=len(tokens) or tokens[i] in values:raise ValueError('Malformed go clock')
        value=int(tokens[i+1])
        if value<0:raise ValueError('Negative clock')
        values[tokens[i]]=value;i+=2
    if 'movetime' in values:
        if len(values)!=1:raise ValueError('Ambiguous movetime and clock')
        available=values['movetime']/1000
        return min(cap,max(0.,available-reserve)),available
    if not {'btime','wtime'}<=values.keys():raise ValueError('Explicit movetime or both side clocks required')
    if values.get('byoyomi',0):raise ValueError('Only Fischer/no-byoyomi clocks supported')
    side='b' if turn==0 else 'w';available=(values[side+'time']+values.get(side+'inc',0))/1000
    return search_seconds(available,reserve=reserve,cap=cap),available


def forced_opening(board):
    require_rule_board(board)
    if len(board.move_stack)>=4:return None
    base=clone_board(board);moves=[m.usi() for m in base.move_stack]
    while base.move_stack:base.pop()
    if base.sfen().split()[:3]!=new_board().sfen().split()[:3]:return None
    if int(base.sfen().split()[-1])!=1:return None
    if tuple(moves)!=OPENING[:len(moves)]:raise ValueError('DR7 initial move history differs')
    return OPENING[len(moves)]


def run(source,target,evaluator,*,cap=2.,reserve=.25,dr7_opening=False,safety_fraction=0.):
    search_seconds(0,reserve=reserve,cap=cap)
    if not math.isfinite(safety_fraction) or not 0<=safety_fraction<=.5:raise ValueError('Invalid safety fraction')
    board=new_board();worker=None;stop=threading.Event();lock=threading.Lock()
    def emit(text):
        with lock:print(text,file=target,flush=True)
    def finish():
        nonlocal worker
        if worker is not None:stop.set();worker.join();worker=None
    def think(position,seconds,available,event):
        started=time.monotonic()
        try:
            opening=forced_opening(position) if dr7_opening else None
            if opening:command=opening;nodes=0;depth=0;score=0
            elif declaration(position,available):command='win';nodes=0;depth=0;score=0
            else:
                if safety_fraction:
                    from .safety import choose_guarded_move
                    result=choose_guarded_move(position,evaluator,seconds=seconds,max_simulations=4096,
                        mate_fraction=.2,safety_fraction=safety_fraction,stop=event).search
                else:
                    result=choose_tactical_move(position,evaluator,seconds=seconds,max_simulations=4096,
                        mate_fraction=.2,mate_plies=5,mate_nodes=4000,stop=event).search
                command=result.move.usi() if result.move else 'resign';nodes=result.nodes;depth=result.depth;score=result.score
            emit(f'info depth {depth} nodes {nodes} score cp {score} time {round((time.monotonic()-started)*1000)}')
            emit('bestmove '+command)
        except Exception as exc:
            emit('info string search error '+type(exc).__name__)
            fallback=next(iter(position.legal_moves),None);emit('bestmove '+(fallback.usi() if fallback else 'resign'))
    try:
        for line in source:
            tokens=line.split()
            if not tokens:continue
            command=tokens[0]
            try:
                if command=='usi':emit('id name shogi-auto policy DR7 offline');emit('id author shogi-auto');emit('usiok')
                elif command=='isready':emit('readyok')
                elif command in ('position','usinewgame'):
                    finish();board=parse_position(tokens[1:]) if command=='position' else new_board()
                elif command=='go':
                    finish();seconds,available=go_budget(tokens,board.turn,reserve=reserve,cap=cap)
                    stop=threading.Event();worker=threading.Thread(target=think,args=(clone_board(board),seconds,available,stop));worker.start()
                elif command in ('stop','gameover'):finish()
                elif command=='quit':break
                elif command=='setoption':emit('info string options are fixed by explicit launch arguments')
            except (ValueError,IndexError) as exc:emit('info string '+str(exc))
    finally:finish()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--max-move-seconds',type=float,default=2.);p.add_argument('--reserve-seconds',type=float,default=.25);p.add_argument('--dr7-opening',action='store_true');p.add_argument('--safety-fraction',type=float,default=0.);a=p.parse_args()
    from .network import Evaluator,load_network
    evaluator=Evaluator(load_network(a.model));run(sys.stdin,sys.stdout,evaluator,cap=a.max_move_seconds,reserve=a.reserve_seconds,dr7_opening=a.dr7_opening,safety_fraction=a.safety_fraction)

if __name__=='__main__':main()
