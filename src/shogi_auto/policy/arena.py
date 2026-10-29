"""CPU arena: equal clocks, paired openings, fixed teacher and unchanged models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import json
from multiprocessing import get_context
from pathlib import Path
import time

import shogi

from ..engine import choose_move
from ..game import Game
from ..learning.arena import create_opening, summarize
from .common import Budget, BudgetExpired, sha256, source_identity, write_json
from .features import feature_key
from .network import Evaluator, load_network
from .search import choose_policy_move
from .teacher import UsiTeacher
from .train import read_data


@lru_cache(maxsize=2)
def evaluator(path):
    return Evaluator(load_network(path))


def play(task):
    row_id, comparison, pair_id, color, opening, directory, binary, eval_dir, config = task
    root = Path(directory)
    initial = str(root / 'initial-policy.json')
    chosen = str(root / 'best-policy.json')
    candidate = initial if comparison == 'initial-vs-teacher' else chosen
    opponent = initial if comparison == 'trained-vs-initial' else None
    strong = comparison in ('trained-vs-teacher','initial-vs-teacher')
    candidate_evaluator = evaluator(candidate)
    opponent_evaluator = evaluator(opponent) if opponent is not None else None
    teacher = UsiTeacher(binary, eval_dir, root / 'teacher-logs' / f'{row_id}.log') if strong else None
    game = Game()
    for usi in opening:game.push(usi)
    times, depths, nodes, turns = [[],[]], [[],[]], [[],[]], [0,0]
    error, reason, winner, illegal, overrun = None, 'move_limit', None, False, False
    try:
        for _ in range(config['max_plies']):
            if game.result:break
            side = int(game.board.turn != color)
            turns[side] += 1
            started = time.monotonic()
            try:
                if side == 0 or opponent is not None:
                    result = choose_policy_move(game.board, candidate_evaluator if side == 0 else opponent_evaluator,
                                                seconds=config['move_seconds'], max_simulations=128)
                    move, depth, count = result.move, result.depth, result.nodes
                elif strong:
                    reply = teacher.analyze(game.board, move_seconds=config['move_seconds'], timeout=.5)
                    if reply['usi'] == 'resign':
                        winner, reason = color, 'teacher_resign';break
                    if reply['usi'] == 'win':
                        raise ValueError('Entering-king declaration requires an independent adjudicator; no fabricated result')
                    move, depth, count = reply['move'], reply['depth'], reply['nodes']
                else:
                    result = choose_move(game.board, seconds=config['move_seconds'], max_nodes=6000)
                    move, depth, count = result.move, result.depth, result.nodes
                elapsed = time.monotonic()-started
                times[side].append(elapsed);depths[side].append(depth);nodes[side].append(count)
                if elapsed > .5:
                    overrun, winner, reason = True, 1-game.board.turn, 'time_forfeit';break
                if move is None or not game.board.is_legal(move):
                    illegal, winner, reason = True, 1-game.board.turn, 'illegal_move';break
                game.push(move.usi())
            except TimeoutError as exc:
                overrun, error, winner, reason = True, str(exc), 1-game.board.turn, 'time_forfeit';break
            except Exception as exc:
                error, winner, reason = type(exc).__name__+': '+str(exc), 1-game.board.turn, 'exception';break
        if game.result:winner, reason = game.result['winner'], game.result['reason']
    finally:
        if teacher:teacher.close()
    outcome = 'draw' if winner is None else 'win' if winner == color else 'loss'
    return {'game_id':row_id,'opponent':comparison,'pair_id':pair_id,'candidate_color':color,
            'opening':opening,'outcome':outcome,'score':.5 if winner is None else float(winner==color),
            'reason':reason,'error':error,'illegal_move':illegal,'time_forfeit':overrun,
            'abnormal':bool(error or illegal or overrun),'moves':[r['usi'] for r in game.history],
            'plies':len(game.history)-len(opening),'turns':turns,
            'mean_search_depth':[sum(v)/len(v) if v else None for v in depths],
            'depth_semantics':'policy=max simulated ply; alpha-beta/teacher=reported completed depth',
            'total_nodes':[sum(v) for v in nodes],
            'node_semantics':'policy=network-guided simulations; other engines=search nodes; not equivalent units',
            'timing_seconds':[{'mean':sum(v)/len(v) if v else None,'max':max(v) if v else None} for v in times]}


def arena(directory, data_directory, binary, eval_dir, config, max_seconds=3600):
    directory = Path(directory)
    arrays, manifest = read_data(data_directory)
    forbidden = set()
    import hashlib
    for x in arrays['x']:
        forbidden.add(min(hashlib.sha256(x[:42].tobytes()).hexdigest(),
                          hashlib.sha256(x[:42,:,::-1].copy().tobytes()).hexdigest()))
    openings = []
    for pair_id in range(config['pairs']):
        for attempt in range(100):
            prefix = create_opening(config['seed'] + pair_id*1000 + attempt, set())
            board = shogi.Board()
            for usi in prefix:board.push_usi(usi)
            if feature_key(board) not in forbidden and prefix not in openings:
                openings.append(prefix);break
        else:raise ValueError('Unable to construct disjoint held-out openings')
    comparisons = ('trained-vs-initial','trained-vs-material','trained-vs-teacher','initial-vs-teacher')
    identity = {'source':source_identity(),'config':config,'data_sha256':manifest['npz_sha256'],
                'models':{name:sha256(directory/name) for name in ('initial-policy.json','best-policy.json')},
                'teacher_binary_sha256':sha256(binary),'teacher_weights_sha256':sha256(Path(eval_dir)/'nn.bin'),
                'openings':openings,'comparisons':list(comparisons)}
    settings_path = directory/'policy-arena-settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != identity:
        raise ValueError('Arena source/config/models changed; use a new directory')
    write_json(settings_path,identity)
    path=directory/'policy-arena-progress.jsonl'
    completed={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);completed[row['game_id']]=row
    tasks=[]
    for comparison in comparisons:
        for pair_id,prefix in enumerate(openings):
            for color in (0,1):
                row_id=f'{comparison}:{pair_id}:{color}'
                if row_id not in completed:
                    tasks.append((row_id,comparison,pair_id,color,prefix,str(directory.resolve()),
                                  str(Path(binary).resolve()),str(Path(eval_dir).resolve()),config))
    budget=Budget(directory/'arena-budget',max_seconds,visible_gpus=0)
    budget.path.parent.mkdir(exist_ok=True)
    status='evaluated'
    try:
        pending=iter(tasks);exhausted=False
        with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool,path.open('a') as stream:
            jobs={}
            while jobs or not exhausted:
                budget.check(reserve=240)
                while len(jobs)<2 and not exhausted:
                    try:task=next(pending)
                    except StopIteration:exhausted=True;break
                    jobs[pool.submit(play,task)]=task[0]
                if not jobs:break
                done,_=wait(jobs,timeout=.5,return_when=FIRST_COMPLETED)
                for future in done:
                    row=future.result();completed[row['game_id']]=row
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    del jobs[future];budget.save()
                    print(f'policy arena: {len(completed)}/{config["pairs"]*8} {row["game_id"]} {row["outcome"]}',flush=True)
    except BudgetExpired:status='paused'
    finally:budget.save()
    games=[r for _,r in sorted(completed.items())]
    summaries={name:summarize([g for g in games if g['opponent']==name],config['pairs']) for name in comparisons}
    report={**identity,'status':status,'games':games,'summaries':summaries,
            'complete':all(x['complete'] for x in summaries.values()),'functional_only':config['functional_only'],
            'improvement_against_prior_baselines_verified':not config['functional_only'] and all(
                summaries[name]['improvement_evidence'] for name in comparisons[:2]),
            'strong_teacher_superiority_verified':not config['functional_only'] and summaries['trained-vs-teacher']['improvement_evidence'],
            'rules_limitations':'No entering-king adjudication or official CSA clocks; declaration stops as an error',
            'budget':json.loads(budget.path.read_text())}
    write_json(directory/'policy-arena-report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--teacher-binary',type=Path,required=True);p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=32);p.add_argument('--max-plies',type=int,default=384)
    p.add_argument('--move-seconds',type=float,default=.1);p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--max-wall-seconds',type=float,default=3600);p.add_argument('--functional-only',action='store_true')
    a=p.parse_args()
    if not (1<=a.pairs<=64 and 16<=a.max_plies<=384 and .01<=a.move_seconds<=.2):raise ValueError('Arena bounds exceeded')
    if a.pairs<30 and not a.functional_only:raise ValueError('At least 30 paired openings are required')
    config={'pairs':a.pairs,'max_plies':a.max_plies,'move_seconds':a.move_seconds,'seed':a.seed,'functional_only':a.functional_only}
    report=arena(a.run_dir,a.data_dir,a.teacher_binary,a.eval_dir,config,a.max_wall_seconds)
    print(json.dumps({k:v for k,v in report.items() if k not in ('games','openings')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
