import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import shogi

from shogi_auto.policy.features import ACTIONS, action_index, encode, legal_actions
from shogi_auto.policy.search import choose_policy_move
from shogi_auto.policy.teacher import UsiTeacher


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy is unavailable')
class PolicyFeaturesTests(unittest.TestCase):
    def test_color_rotation_and_hands(self):
        import numpy as np
        black = shogi.Board()
        white = shogi.Board();white.turn = shogi.WHITE
        np.testing.assert_array_equal(encode(black), encode(white))
        self.assertEqual(action_index(black, shogi.Move.from_usi('7g7f')),
                         action_index(white, shogi.Move.from_usi('3c3d')))
        board = shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b 5Pr 1')
        self.assertAlmostEqual(float(encode(board)[28, 0, 0]), 5/18)
        self.assertAlmostEqual(float(encode(board)[41, 0, 0]), .5)
        label = action_index(board, shogi.Move.from_usi('P*5e'))
        self.assertEqual(label // 81, 20)

    def test_legal_action_labels_are_unique(self):
        rng = random.Random(8001)
        board = shogi.Board()
        encountered_drop = encountered_promotion = False
        for _ in range(150):
            actions = legal_actions(board)
            self.assertEqual(len(actions), sum(1 for _ in board.legal_moves))
            self.assertTrue(all(0 <= index < ACTIONS for index in actions))
            if not actions or board.is_fourfold_repetition():
                break
            move = rng.choice(list(actions.values()))
            encountered_drop |= move.from_square is None
            encountered_promotion |= move.promotion
            board.push(move)
        self.assertTrue(encountered_drop)
        self.assertTrue(encountered_promotion)

    def test_puct_value_sign_and_mate(self):
        board = shogi.Board('4k4/9/4R4/9/9/9/9/9/4K4 b G 1')
        before = board.sfen()
        target = shogi.Move.from_usi('G*5b')
        def evaluate(position):
            legal = [m for m in position.legal_moves]
            if not legal:
                return {}, -1.0
            return {move: (.99 if move == target else .01/len(legal)) for move in legal}, 0.0
        result = choose_policy_move(board, evaluate, seconds=2, max_simulations=32)
        self.assertEqual(result.move, target)
        self.assertGreater(result.score, 0)
        self.assertEqual(board.sfen(), before)


class TeacherProtocolTests(unittest.TestCase):
    def test_handshake_exact_scores_and_legal_moves_without_external_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'fixture-engine'
            engine.write_text('''#!/usr/bin/env python3
import sys
for line in sys.stdin:
    command = line.strip()
    if command == 'usi':
        print('id name Test fixture; no actual shogi search')
        for name in ('EvalDir', 'Threads', 'USI_Hash', 'BookFile', 'FV_SCALE'):
            print('option name '+name+' type string default x')
        print('usiok', flush=True)
    elif command == 'isready': print('readyok', flush=True)
    elif command.startswith('go '):
        print('info depth 5 score cp 123 nodes 1025 pv 7g7f')
        print('bestmove 7g7f', flush=True)
    elif command == 'quit': break
''')
            engine.chmod(0o700)
            with UsiTeacher(engine, root, root / 'usi.log') as teacher:
                result = teacher.analyze(shogi.Board(), nodes=1024)
                self.assertEqual(result['usi'], '7g7f')
                self.assertEqual((result['score_kind'], result['score'], result['depth'], result['nodes']),
                                 ('cp', 123, 5, 1025))
                with self.assertRaises(ValueError):teacher.send('isready\nquit')


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'optional PyTorch/NumPy are unavailable')
class PolicyNetworkTests(unittest.TestCase):
    def test_optimizer_resume_matches_uninterrupted_training(self):
        import numpy as np
        from unittest.mock import patch
        from shogi_auto.policy.common import BudgetExpired, sha256, write_json
        from shogi_auto.policy.train import train
        rng = np.random.default_rng(88)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);data = root/'data';data.mkdir()
            x = rng.random((6,43,9,9),dtype=np.float32)
            legal = np.zeros((6,ACTIONS),dtype=np.bool_);legal[:,:2] = True
            np.savez_compressed(data/'data.npz',x=x,legal=legal,policy=np.zeros(6,dtype=np.int64),
                                value=np.zeros(6,dtype=np.float32),split=np.array(['train']*4+['validation','test']),
                                game_id=np.arange(6,dtype=np.int64))
            write_json(data/'data-manifest.json',{'npz_sha256':sha256(data/'data.npz'),
                'counts':{'train':4,'validation':1,'test':1},'teacher_binary_sha256':'0'*64,
                'teacher_weights_sha256':'1'*64,'teacher_identity':['unit-test fixture; no real teacher']})
            class StopAfterEpoch:
                def __init__(self, destination, *args, **kwargs):
                    self.path=Path(destination)/'budget.json';self.saved=0
                def check(self, **kwargs):
                    if self.saved:raise BudgetExpired('unit-test pause after first saved epoch')
                def remaining(self):return 100
                def save(self):
                    self.saved+=1;write_json(self.path,{'used_wall_seconds':0,'limit_wall_seconds':60})
            with patch('shogi_auto.policy.train.Budget',StopAfterEpoch):
                first=train(root/'resumed',data,epochs=3,batch_size=2,max_seconds=60,functional_only=True)
            self.assertEqual(first['status'],'paused');self.assertEqual(first['completed_epochs'],1)
            resumed=train(root/'resumed',data,epochs=3,batch_size=2,max_seconds=60,functional_only=True)
            full=train(root/'full',data,epochs=3,batch_size=2,max_seconds=60,functional_only=True)
            self.assertEqual(resumed['selected_weights_sha256'],full['selected_weights_sha256'])
            self.assertEqual(resumed['logs'],full['logs'])
            self.assertEqual(resumed['cuda_optimizer_steps_total'],0)

    def test_training_export_and_cpu_inference_agree(self):
        import torch
        from shogi_auto.policy.network import build_network, export_network, load_network, Evaluator
        torch.set_num_threads(1);torch.manual_seed(93)
        model = build_network()
        board = shogi.Board()
        x = torch.from_numpy(encode(board))[None]
        target = action_index(board, shogi.Move.from_usi('7g7f'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        logits, value = model(x)
        mask = torch.zeros(ACTIONS, dtype=torch.bool);mask[list(legal_actions(board))] = True
        loss = torch.nn.functional.cross_entropy(logits.masked_fill(~mask, -1e9), torch.tensor([target])) + value.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            export_network(path, model, {'purpose': 'unit_test_only'})
            loaded = load_network(path)
            with torch.no_grad():
                expected, expected_value = model(x);actual, actual_value = loaded(x)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)
            priors, estimate = Evaluator(loaded)(board)
            self.assertAlmostEqual(sum(priors.values()), 1, places=6)
            self.assertTrue(-1 <= estimate <= 1)
            self.assertEqual(set(priors), set(legal_actions(board).values()))
            malformed = json.loads(path.read_text());malformed['state']['stem.weight'] = []
            path.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):load_network(path)


if __name__ == '__main__':
    unittest.main()
