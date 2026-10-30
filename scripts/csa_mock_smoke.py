"""One real loopback TCP game with dummy identities; no external server CLI."""
import argparse
import json
import math
from pathlib import Path
import socket
import threading
import time

import shogi

from shogi_auto.csa import LineTransport,Session
from shogi_auto.engine import choose_move
from shogi_auto.policy.common import sha256,write_json
from shogi_auto.policy.tournament import CSA_KINDS,TournamentGame,declaration,parse_csa_move


def game_summary(color):
    lines=['BEGIN Game_Summary','Protocol_Version:1.2','Protocol_Mode:Server','Format:Shogi 1.0',
           'Declaration:Jishogi 1.1','Game_ID:dr7-loopback','Name+:local-black','Name-:local-white',
           'Your_Turn:'+('+' if color==0 else '-'),'To_Move:+','Max_Moves:512']
    for sign,total in [('+',180),('-',600)]:
        lines += ['BEGIN Time'+sign,'Time_Unit:1sec','Total_Time:'+str(total),'Increment:2',
                  'Least_Time_Per_Move:0','Time_Roundup:NO','END Time'+sign]
    lines.append('BEGIN Position');board=shogi.Board()
    for rank in range(9):
        row=f'P{rank+1}'
        for square in range(rank*9,rank*9+9):
            p=board.piece_at(square)
            row+=('+' if p.color==0 else '-')+CSA_KINDS[p.piece_type] if p else ' * '
        lines.append(row)
    return lines+['P+','P-','+','END Position','END Game_Summary']


def run(destination,max_seconds=60):
    started=time.monotonic();errors=[];clients={};sessions={};wire_moves=[];server_result={}
    listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(2);listener.settimeout(5)
    address=listener.getsockname();peers={};game=TournamentGame()
    def broadcast(line):
        for transport in peers.values():transport.send(line)
    def server():
        try:
            for _ in (0,1):
                connection,_=listener.accept();transport=LineTransport(connection,5)
                login=transport.receive()
                if login not in ('LOGIN fixture0 mock-only','LOGIN fixture1 mock-only'):raise ValueError('Wrong dummy login')
                color=int(login.split()[1][-1]);peers[color]=transport
                transport.send('LOGIN:fixture'+str(color)+' OK')
                for line in game_summary(color):transport.send(line)
            for transport in peers.values():
                if transport.receive()!='AGREE dr7-loopback':raise ValueError('Missing Agree')
            game.clock.begin(0);turn_started=time.monotonic();broadcast('START:dr7-loopback');pending_mate=None
            while True:
                if time.monotonic()-started>max_seconds:raise TimeoutError('Loopback smoke budget reached')
                side=game.game.board.turn;command=peers[side].receive();elapsed=time.monotonic()-turn_started
                spent=math.floor(elapsed)
                if pending_mate:
                    if command!='%TORYO':raise ValueError('Mated client did not resign')
                    game.clock.elapse(side,elapsed);result=pending_mate;reason='#RESIGN'
                else:
                    usi=command if command.startswith('%') else parse_csa_move(game.game.board,command)[0]
                    result=game.play(usi,elapsed)
                    reason={'512_move_limit':'#MAX_MOVES','千日手':'#SENNICHITE','連続王手の千日手':'#OUTE_SENNICHITE',
                            'resign':'#RESIGN','entering_king':'#JISHOGI','false_declaration':'#ILLEGAL_MOVE',
                            'illegal_move':'#ILLEGAL_MOVE','opening_violation':'#ILLEGAL_MOVE','time_forfeit':'#TIME_UP'}.get(result['reason']) if result else None
                broadcast(command+',T'+str(spent));wire_moves.append({'side':side,'command':command,'spent':spent})
                if result and result['reason'] in ('詰み','合法手なし') and not pending_mate:
                    pending_mate=result;game.clock.begin(game.game.board.turn);turn_started=time.monotonic();continue
                if result:
                    if reason is None:raise ValueError('Unmapped ending')
                    broadcast(reason)
                    for color,transport in peers.items():
                        transport.send(('#CENSORED' if reason=='#MAX_MOVES' else '#DRAW') if result['winner'] is None
                                       else '#WIN' if result['winner']==color else '#LOSE')
                    server_result.update(result);break
                game.clock.begin(game.game.board.turn);turn_started=time.monotonic()
            for transport in peers.values():
                if transport.receive()!='LOGOUT':raise ValueError('Missing logout')
                transport.send('LOGOUT:completed')
        except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
        finally:
            for transport in peers.values():transport.close()
            listener.close()
    def client(color):
        connection=socket.create_connection(address,timeout=5);session=Session(connection,5,max_receive_seconds=5);sessions[color]=session
        try:
            if not session.start('fixture'+str(color),'mock-only'):raise ValueError('Mock handshake failed')
            def player(board,remaining):
                if declaration(board,remaining):return '%KACHI'
                result=choose_move(board,seconds=.01,max_nodes=256)
                return result.move.usi() if result.move else '%TORYO'
            clients[color]=session.run(player)
            if session.state=='finished':session.logout()
        except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
        finally:session.close()
    referee=threading.Thread(target=server,daemon=True);referee.start()
    workers=[threading.Thread(target=client,args=(color,),daemon=True) for color in (0,1)]
    for worker in workers:worker.start()
    for worker in workers:worker.join(timeout=max_seconds+5)
    referee.join(timeout=5)
    if any(worker.is_alive() for worker in workers) or referee.is_alive():errors.append('Unterminated mock worker')
    if errors:raise RuntimeError('; '.join(errors))
    if len(clients)!=2 or not all(value['server_confirmed'] for value in clients.values()):raise ValueError('Incomplete loopback match')
    for color,session in sessions.items():
        if session.clock.remaining!=game.clock.remaining:raise ValueError('Server/client clock divergence')
        if len(session.game.history)!=len(game.game.history):raise ValueError('Server/client board divergence')
        if session.game.board.sfen()!=game.game.board.sfen():raise ValueError('Different terminal positions')
        if 'mock-only' in str(session.transport.trace):raise ValueError('Password leaked into trace')
    report={'status':'passed','transport':'TCP 127.0.0.1 ephemeral port','real_tournament_server_connected':False,
            'wall_seconds':time.monotonic()-started,'server_result':server_result,'client_results':clients,
            'plies':len(game.game.history),'moves':[x['usi'] for x in game.game.history],'wire_moves':wire_moves,
            'clocks_remaining':game.clock.remaining,'server_client_positions_clocks_verified':True,
            'client_sha256':sha256(Path(__file__).resolve().parents[1]/'src/shogi_auto/csa.py'),
            'rules_sha256':sha256(Path(__file__).resolve().parents[1]/'src/shogi_auto/policy/tournament.py'),
            'purpose':'Protocol integration smoke only; no learning or strength comparison'}
    write_json(destination,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/csa-loopback-smoke.json'))
    a=p.parse_args();r=run(a.output);print(json.dumps({k:v for k,v in r.items() if k not in ('moves','wire_moves')},ensure_ascii=False,indent=2))
