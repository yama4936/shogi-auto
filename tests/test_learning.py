import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import shogi

from shogi_auto.engine import evaluate, choose_move
from shogi_auto.learning.arena import summarize
from shogi_auto.learning.data import feature_key, split_for_game
from shogi_auto.learning.features import FEATURES, LinearEvaluator, encode, export_model


class ModelTests(unittest.TestCase):
    def test_zero_model_preserves_baseline_and_turn_antisymmetry(self):
        board = shogi.Board()
        for move in ('7g7f', '3c3d', '8h2b+', '3a2b', 'B*4e'):
            board.push_usi(move)
        vector = encode(board)
        self.assertEqual(len(vector), FEATURES)
        model = LinearEvaluator([0.0] * FEATURES)
        self.assertEqual(model(board), evaluate(board))
        board.turn = 1 - board.turn
        self.assertEqual(encode(board), [-value for value in vector])

    def test_model_validation_export_and_correction_bound(self):
        with self.assertRaises(ValueError):
            LinearEvaluator([float('nan')] * FEATURES)
        board = shogi.Board(); board.push_usi('7g7f')
        vector = encode(board)
        weights = [value * 100 for value in vector]
        model = LinearEvaluator(weights)
        self.assertEqual(model(board) - evaluate(board), 600)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            export_model(path, weights, {'test': True})
            self.assertEqual(LinearEvaluator.load(path)(board), model(board))
        before = board.sfen()
        move = choose_move(board, seconds=0.01, evaluator=model).move
        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.sfen(), before)

    def test_mirror_deduplication_and_game_split(self):
        vector = [0.0] * FEATURES
        vector[2 * 81 + 7] = 1
        mirrored = [0.0] * FEATURES
        mirrored[2 * 81 + 1] = 1
        self.assertEqual(feature_key(vector), feature_key(mirrored))
        self.assertEqual([split_for_game(i) for i in range(3)], ['test', 'validation', 'train'])

    def test_small_perfect_sample_does_not_claim_improvement(self):
        games = [{'pair_id': i // 2, 'outcome': 'win', 'score': 1.0, 'reason': '詰み',
                  'abnormal': False, 'illegal_move': False, 'time_forfeit': False} for i in range(8)]
        result = summarize(games, 4)
        self.assertFalse(result['improvement_evidence'])
        self.assertLess(result['score_95pct_hoeffding_bound'][0], 0.5)
        limited = [{**game, 'reason': 'move_limit'} for game in games]
        self.assertEqual(summarize(limited, 4)['move_limit_games'], 8)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'optional learning dependencies')
class TrainingTests(unittest.TestCase):
    def test_checkpoint_resume_and_data_identity_guard(self):
        import numpy as np
        from shogi_auto.learning.common import Budget
        from shogi_auto.learning.train import train_model
        from shogi_auto.learning.run import DEFAULTS
        rng = np.random.default_rng(91)
        x = np.zeros((40, FEATURES), dtype=np.float32)
        x[:, :4] = rng.normal(size=(40, 4))
        arrays = {'x': x, 'y': x[:, 0] * 0.1, 'split': np.array(['train'] * 24 + ['validation'] * 8 + ['test'] * 8)}
        config = {**DEFAULTS, 'epochs': 4}
        with tempfile.TemporaryDirectory() as directory:
            first = train_model(directory, arrays, {'corpus_sha256': 'owned-fixture'}, config, Budget(directory, 60))
            model = (Path(directory) / 'best-model.json').read_bytes()
            second = train_model(directory, arrays, {'corpus_sha256': 'owned-fixture'}, config, Budget(directory, 60))
            self.assertEqual(first['completed_epochs'], second['completed_epochs'])
            self.assertEqual((Path(directory) / 'best-model.json').read_bytes(), model)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                train_model(directory, arrays, {'corpus_sha256': 'different-data'}, config, Budget(directory, 60))

    def test_interrupted_training_resumes_same_optimizer_trajectory(self):
        import numpy as np
        from shogi_auto.learning.common import Budget, BudgetExpired
        from shogi_auto.learning.train import train_model
        from shogi_auto.learning.run import DEFAULTS
        class InterruptBudget(Budget):
            calls = 0
            def check(self, reserve=10):
                self.calls += 1
                if self.calls == 3:
                    raise BudgetExpired('simulated interruption after two saved epochs')
                return super().check(reserve)
        rng = np.random.default_rng(37)
        x = np.zeros((32, FEATURES), dtype=np.float32)
        x[:, :3] = rng.normal(size=(32, 3))
        arrays = {'x': x, 'y': x[:, 0] * .1, 'split': np.array(['train'] * 16 + ['validation'] * 8 + ['test'] * 8)}
        config = {**DEFAULTS, 'epochs': 6}
        with tempfile.TemporaryDirectory() as interrupted, tempfile.TemporaryDirectory() as fresh:
            with self.assertRaises(BudgetExpired):
                train_model(interrupted, arrays, {'corpus_sha256': 'resume-fixture'}, config, InterruptBudget(interrupted, 60))
            resumed = train_model(interrupted, arrays, {'corpus_sha256': 'resume-fixture'}, config, Budget(interrupted, 60))
            uninterrupted = train_model(fresh, arrays, {'corpus_sha256': 'resume-fixture'}, config, Budget(fresh, 60))
            resumed_weights = json.loads((Path(interrupted) / 'best-model.json').read_text())['weights']
            fresh_weights = json.loads((Path(fresh) / 'best-model.json').read_text())['weights']
            self.assertEqual(resumed_weights, fresh_weights)
            self.assertEqual(resumed['logs'], uninterrupted['logs'])


if __name__ == '__main__':
    unittest.main()
