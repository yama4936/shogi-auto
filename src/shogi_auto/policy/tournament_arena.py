"""Small local full-rule games, reported separately from short screening."""
import argparse
import json
from pathlib import Path
import time

from ..engine import choose_move
from .common import Budget,BudgetExpired,sha256,source_identity,write_json
from .network import Evaluator,load_network
from .search import choose_policy_move
from .teacher import UsiTeacher
from .tournament import OPENING,RULES_URL,RULES_SHA256,TournamentGame,declaration


def evaluate(directory,binary,eval_dir,max_seconds=1800):
    directory=Path(directory);out=directory/'tournament-local';out.mkdir(exist_ok=True)
    paths={k:directory/name for k,name in [('candidate','best-policy.json'),('before','initial-policy.json')]}
    evaluators={name:Evaluator(load_network(path)) for name,path in paths.items()}
    identity={'source':source_identity(),'models':{k:sha256(p) for k,p in paths.items()},
              'teacher_binary_sha256':sha256(binary),'teacher_weights_sha256':sha256(Path(eval_dir)/'nn.bin'),
              'rules_url':RULES_URL,'rules_sha256':RULES_SHA256,'move_search_cap_seconds':2.0,
              'communication_mode':'local in-process; no real CSA transport or server authentication tested'}
    settings=out/'settings.json'
    if settings.exists() and json.loads(settings.read_text())!=identity:raise ValueError('Local-rule arena identity changed')
    write_json(settings,identity);budget=Budget(out,max_seconds,0)
    path=out/'games.jsonl';games=[]
    if path.exists():games=[json.loads(line) for line in path.read_text().splitlines()]
    completed={row['id'] for row in games};status='evaluated'
    try:
        for opponent in ('before','material','teacher'):
            for color in (0,1):
                row_id=f'{opponent}:{color}'
                if row_id in completed:continue
                budget.check(reserve=20)
                teacher=UsiTeacher(binary,eval_dir,out/f'teacher-{color}.log') if opponent=='teacher' else None
                game=TournamentGame();timings=[[],[]];error=None
                try:
                    while not game.result:
                        budget.check(reserve=10)
                        board=game.game.board;side=board.turn;game.clock.begin(side);started=time.monotonic()
                        if len(game.game.history)<4:command=OPENING[len(game.game.history)]
                        elif declaration(board,game.clock.remaining[side]):command='%KACHI'
                        else:
                            seconds=min(2.,max(.01,game.clock.remaining[side]/40))
                            if side==color or opponent=='before':
                                result=choose_policy_move(board,evaluators['candidate' if side==color else 'before'],
                                                          seconds=seconds,max_simulations=128)
                                command=result.move.usi() if result.move else '%TORYO'
                            elif opponent=='material':
                                result=choose_move(board,seconds=seconds,max_nodes=6000)
                                command=result.move.usi() if result.move else '%TORYO'
                            else:
                                result=teacher.analyze(board,move_seconds=seconds,timeout=seconds+2)
                                command='%KACHI' if result['usi']=='win' else '%TORYO' if result['usi']=='resign' else result['usi']
                        elapsed=time.monotonic()-started;timings[side].append(elapsed)
                        game.play(command,elapsed)
                except BudgetExpired:raise
                except Exception as exc:
                    error=type(exc).__name__+': '+str(exc)
                    game.finish(1-game.game.board.turn,'exception')
                finally:
                    if teacher:teacher.close()
                row={'id':row_id,'opponent':opponent,'candidate_color':color,**game.result,'error':error,
                     'normal_score':game.result['normal_scores'][color],
                     'tournament_score':game.result['tournament_scores'][color],
                     'moves':[x['usi'] for x in game.game.history],'plies':len(game.game.history),
                     'clocks_remaining':game.clock.remaining,'move_seconds':timings}
                games.append(row)
                with path.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                budget.save();print(f'local DR7 arena: {len(games)}/6 {row_id} {row["reason"]}',flush=True)
    except BudgetExpired:status='paused'
    finally:budget.save()
    report={**identity,'status':status,'complete':len(games)==6,'games':games,'budget':json.loads(budget.path.read_text()),
            'purpose':'Small rule/clock/declaration integration matches; not independent-opening strength evidence',
            'real_csa_connection_verified':False,'communication_failures_external':'not tested',
            'time_forfeits':sum(g['reason']=='time_forfeit' for g in games),
            'illegal_moves':sum(g['reason']=='illegal_move' for g in games),
            'false_declarations':sum(g['reason']=='false_declaration' for g in games),
            'abnormal_terminations':sum(bool(g['error']) for g in games)}
    write_json(out/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--teacher-binary',type=Path,required=True);p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=1800)
    a=p.parse_args();print(json.dumps(evaluate(a.run_dir,a.teacher_binary,a.eval_dir,a.max_wall_seconds),ensure_ascii=False))
