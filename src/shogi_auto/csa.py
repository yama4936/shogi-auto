"""Bounded CSA v1.2.1 transport/session, tested locally; no remote-login CLI.

Primary protocol: https://www.computer-shogi.org/protocol/tcp_ip_server_121.html
The document version is 1.2.1; its advertised Protocol_Version remains 1.2.
"""
from dataclasses import dataclass
import re
import socket
import time

import shogi
import shogi.CSA

from .game import Game
from .policy.tournament import Clock, OPENING, csa_move, declaration, parse_csa_move


class ProtocolError(ValueError):
    pass


class CommunicationError(ConnectionError):
    pass


class LineTransport:
    def __init__(self, connection, timeout=3):
        self.connection=connection;self.timeout=timeout;self.buffer=bytearray();self.trace=[]

    @staticmethod
    def valid(line):
        if any(ord(c)<32 or ord(c)>127 for c in line):raise ProtocolError('Non-CSA control or non-ASCII byte')

    def send(self,line):
        self.valid(line)
        self.trace.append(('send','LOGIN [redacted]' if line.startswith('LOGIN ') else line))
        try:self.connection.sendall((line+'\n').encode('ascii'))
        except OSError as exc:raise CommunicationError('CSA send failed') from exc

    def receive(self):
        deadline=time.monotonic()+self.timeout
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise CommunicationError('CSA receive deadline reached')
            self.connection.settimeout(remaining)
            try:raw=self.connection.recv(4096)
            except (OSError,socket.timeout) as exc:raise CommunicationError('CSA receive failed or timed out') from exc
            if not raw:raise CommunicationError('CSA connection closed')
            self.buffer.extend(raw)
            if len(self.buffer)>32768:raise ProtocolError('CSA frame buffer limit exceeded')
        raw,_,rest=self.buffer.partition(b'\n');self.buffer=bytearray(rest)
        if len(raw)>8192:raise ProtocolError('CSA line limit exceeded')
        try:line=raw.decode('ascii')
        except UnicodeDecodeError as exc:raise ProtocolError('Non-ASCII CSA frame') from exc
        self.valid(line);self.trace.append(('receive','LOGIN [redacted]' if line.startswith('LOGIN ') else line));return line

    def close(self):
        try:self.connection.close()
        except OSError:pass


@dataclass
class Summary:
    fields: dict
    times: dict
    position: list

    @classmethod
    def read(cls,transport,on_field=None):
        if transport.receive()!='BEGIN Game_Summary':raise ProtocolError('Missing Game_Summary')
        fields,times,position={}, {}, []
        stack=['Game_Summary']
        for _ in range(1024):
            line=transport.receive()
            if line.startswith('BEGIN '):
                name=line[6:]
                if len(stack)!=1 or name not in ('Time','Time+','Time-','Position'):raise ProtocolError('Unexpected summary nesting')
                stack.append(name)
                if name!='Position':
                    if name in times:raise ProtocolError('Duplicate time section')
                    times[name]={}
            elif line.startswith('END '):
                if not stack or line[4:]!=stack[-1]:raise ProtocolError('Mismatched summary ending')
                stack.pop()
                if not stack:return cls(fields,times,position)
            elif stack[-1]=='Position':position.append(line)
            else:
                if ':' not in line:raise ProtocolError('Malformed summary field')
                key,value=line.split(':',1);target=fields if len(stack)==1 else times[stack[-1]]
                if key in target:raise ProtocolError('Duplicate summary field')
                target[key]=value
                if len(stack)==1 and on_field:on_field(key,value)
        raise ProtocolError('Summary line limit exceeded')

    def validate_dr7(self):
        f=self.fields
        if f.get('Protocol_Version') not in ('1.2','1.2.1') or f.get('Protocol_Mode','Server')!='Server':
            raise ProtocolError('Unsupported CSA mode/version')
        if (f.get('Format')!='Shogi 1.0' or f.get('Declaration')!='Jishogi 1.1' or
            not f.get('Name+') or not f.get('Name-') or f.get('Your_Turn') not in ('+','-') or
            f.get('To_Move') not in ('+','-') or f.get('Rematch_On_Draw','NO')!='NO' or f.get('Max_Moves')!='512'):
            raise ProtocolError('Incomplete or non-DR7 game conditions')
        for color,total in [('+','180'),('-','600')]:
            t=self.times.get('Time'+color,self.times.get('Time',{}))
            expected={'Time_Unit':'1sec','Total_Time':total,'Increment':'2','Byoyomi':'0',
                      'Delay':'0','Least_Time_Per_Move':'0','Time_Roundup':'NO'}
            for key,value in expected.items():
                default='1sec' if key=='Time_Unit' else 'NO' if key=='Time_Roundup' else '0'
                if t.get(key,default)!=value:raise ProtocolError('Unsupported DR7 time condition: '+key)
        ranks=[line for line in self.position if re.match(r'^P[1-9]',line)]
        if len(ranks)!=9 or {line[:2] for line in ranks}!={f'P{i}' for i in range(1,10)} or any(len(v)!=29 for v in ranks):
            raise ProtocolError('CSA requires explicit nine rank rows')
        if any(line.startswith('PI') or 'AL' in line for line in self.position):raise ProtocolError('Unsupported compact position')

    def restore(self):
        base=[];moves=[];turn=None
        for line in self.position:
            if line.startswith('P'):
                if turn is not None:raise ProtocolError('Piece rows after turn marker')
                base.append(line)
            elif line in ('+','-'):
                if turn is not None:raise ProtocolError('Duplicate position turn')
                turn=line
            elif line.startswith(('+','-')):moves.append(line)
            elif line:raise ProtocolError('Unsupported position entry')
        if turn is None:raise ProtocolError('Missing initial turn')
        parsed=shogi.CSA.Parser.parse_str('\n'.join(base+[turn])+'\n')[0]
        game=Game(sfen=parsed['sfen']);clock=Clock()
        for line in moves:
            usi,spent=parse_csa_move(game.board,line)
            if spent is None:raise ProtocolError('Missing history time')
            side=game.board.turn
            if len(game.history)<4 and usi!=OPENING[len(game.history)]:raise ProtocolError('Wrong initial DR7 moves')
            if not clock.elapse(side,spent):raise ProtocolError('Expired history clock')
            game.push(usi);clock.moved(side)
        if len(game.history)>512:raise ProtocolError('Too many restored moves')
        if ('+' if game.board.turn==0 else '-')!=self.fields['To_Move']:raise ProtocolError('To_Move differs from restored position')
        return game,clock


class Session:
    def __init__(self,connection,timeout=3,max_receive_seconds=None):
        self.transport=LineTransport(connection,timeout);self.state='connected';self.summary=None
        self.game=None;self.clock=None;self.color=None;self.pending=None;self.reason=None;self.result=None
        self.waiting_result=False;self.events=[]
        self.last_special=None
        self.base_timeout=timeout;self.max_receive_seconds=max_receive_seconds

    def login(self,username,password):
        if self.state!='connected' or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',username):raise ProtocolError('Invalid login identity/state')
        if not 1<=len(password)<=32 or any(not 33<=ord(c)<=126 for c in password):raise ProtocolError('Invalid password encoding')
        self.transport.send('LOGIN '+username+' '+password)
        if self.transport.receive()!='LOGIN:'+username+' OK':raise ProtocolError('CSA login rejected or identity differs')
        self.state='waiting'

    def negotiate(self):
        if self.state!='waiting':raise ProtocolError('Cannot negotiate now')
        self.state='summarizing'
        def field(key,value):
            if key=='Your_Turn' and value in ('+','-'):self.color=0 if value=='+' else 1
        self.summary=Summary.read(self.transport,field)
        own=self.summary.fields.get('Your_Turn');self.color=0 if own=='+' else 1 if own=='-' else None
        self.summary.validate_dr7();self.game,self.clock=self.summary.restore()
        game_id=self.summary.fields.get('Game_ID','')
        self.transport.send('AGREE'+(' '+game_id if game_id else ''));self.state='agreed'
        line=self.transport.receive()
        if line!='START:'+game_id:raise ProtocolError('CSA did not start agreed game')
        self.clock.begin(self.game.board.turn);self.state='playing'

    def fail(self,exc):
        phase=self.state;self.state='failed'
        self.result={'outcome':'local_failure','reason':type(exc).__name__,'detail':str(exc),
            'phase':phase,'server_confirmed':False,'normal_score':0 if self.color is not None else None,
            'tournament_score':0 if self.color is not None else None,'automatic_reconnect':False}
        return self.result

    def start(self,username,password):
        try:self.login(username,password);self.negotiate()
        except (CommunicationError,ProtocolError,ValueError) as exc:
            self.fail(exc);return False
        return True

    def send_decision(self,usi):
        if self.state!='playing' or self.pending is not None or self.game.board.turn!=self.color or self.waiting_result:
            raise ProtocolError('Not allowed to send a decision now')
        if len(self.game.history)<4 and usi!=OPENING[len(self.game.history)]:raise ProtocolError('Required initial move differs')
        if usi=='%KACHI':
            if not declaration(self.game.board,self.clock.remaining[self.color]):raise ProtocolError('Invalid local entering-king declaration')
            line=usi
        elif usi=='%TORYO':line=usi
        else:line=csa_move(self.game.board,usi)
        self.transport.send(line);self.pending=line

    def receive_event(self):
        available=self.clock.remaining[self.game.board.turn]
        wait=max(self.base_timeout,available+3)
        self.transport.timeout=min(wait,self.max_receive_seconds) if self.max_receive_seconds is not None else wait
        line=self.transport.receive()
        if not line:return
        if line.startswith('#'):
            if line in ('#WIN','#LOSE','#DRAW','#CENSORED'):
                if self.reason is None:raise ProtocolError('Outcome without reason')
                draw=line in ('#DRAW','#CENSORED')
                if line=='#CENSORED' and self.reason!='#MAX_MOVES':raise ProtocolError('Unexpected censorship')
                outcome='draw' if draw else 'win' if line=='#WIN' else 'loss'
                self.result={'outcome':outcome,'reason':self.reason,'server_confirmed':True,
                    'normal_score':.5 if draw else float(outcome=='win'),
                    'tournament_score':(.4 if self.color==0 else .6) if draw else float(outcome=='win')}
                self.state='finished';return
            if line=='#CHUDAN':
                self.result={'outcome':'interrupted','reason':line,'server_confirmed':True,
                             'normal_score':None,'tournament_score':None};self.state='finished';return
            if line not in ('#RESIGN','#JISHOGI','#SENNICHITE','#OUTE_SENNICHITE','#ILLEGAL_MOVE','#TIME_UP','#MAX_MOVES','#ILLEGAL_ACTION'):
                raise ProtocolError('Unknown CSA ending')
            if line=='#MAX_MOVES' and len(self.game.history)!=512:raise ProtocolError('Premature move limit')
            if line=='#JISHOGI' and self.last_special!='%KACHI':raise ProtocolError('Declaration result without declaration')
            if line=='#RESIGN' and self.last_special!='%TORYO':raise ProtocolError('Resignation result without resignation')
            if line=='#SENNICHITE' and not self.game.board.is_fourfold_repetition():raise ProtocolError('Premature repetition result')
            self.reason=line;self.waiting_result=True;return
        side=self.game.board.turn
        command=line.split(',')[0]
        match=re.search(r',T(\d+)$',line)
        if match is None:raise ProtocolError('Missing server move time')
        spent=int(match[1])
        if side==self.color and self.pending!=command:raise ProtocolError('Own move echo differs from pending move')
        expired=not self.clock.elapse(side,spent)
        if expired:self.events.append({'server_time_exhausted':side,'spent':spent})
        if command in ('%KACHI','%TORYO'):
            self.last_special=command
            self.events.append({'command':command,'spent':spent,'local_declaration_valid':
                declaration(self.game.board,self.clock.remaining[side]) if command=='%KACHI' else None})
            self.waiting_result=True
        else:
            try:usi,_=parse_csa_move(self.game.board,line)
            except ValueError:
                self.waiting_result=True;self.events.append({'invalid_server_move':line,'spent':spent})
            else:
                if len(self.game.history)<4 and usi!=OPENING[len(self.game.history)]:raise ProtocolError('Server initial move differs')
                self.game.push(usi);self.clock.moved(side)
                terminal=self.game.result
                repetition=terminal and terminal['reason'] in ('千日手','連続王手の千日手')
                mate=terminal and terminal['reason'] in ('詰み','合法手なし')
                self.waiting_result=bool(expired or repetition or len(self.game.history)>=512 or
                                         (mate and self.game.board.turn!=self.color))
                if not self.waiting_result:self.clock.begin(self.game.board.turn)
        if side==self.color:self.pending=None

    def run(self,player):
        if self.state!='playing':raise ProtocolError('Game not started')
        try:
            while self.state=='playing':
                if self.game.board.turn==self.color and self.pending is None and not self.waiting_result:
                    if self.game.result and self.game.result['reason'] in ('詰み','合法手なし'):usi='%TORYO'
                    else:usi=OPENING[len(self.game.history)] if len(self.game.history)<4 else player(self.game.board,self.clock.remaining[self.color])
                    self.send_decision(usi)
                self.receive_event()
        except (CommunicationError,ProtocolError,ValueError) as exc:
            self.fail(exc)
        return self.result

    def logout(self):
        if self.state not in ('waiting','finished'):raise ProtocolError('Cannot logout during a game')
        self.transport.send('LOGOUT')
        if self.transport.receive()!='LOGOUT:completed':raise ProtocolError('Logout not confirmed')
        self.state='closed';self.close()

    def close(self):self.transport.close()
