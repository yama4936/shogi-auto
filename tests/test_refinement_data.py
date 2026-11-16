from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shogi_auto.learning.data import split_for_game
from shogi_auto.policy.data import trajectory_actor


class RefinementDataTests(unittest.TestCase):
    def test_student_and_teacher_actors_occur_in_every_whole_game_split(self):
        config = {'seed': 20261106, 'seed_namespace': 'tactical-refinement-v1'}
        counts = {split: Counter() for split in ('train', 'validation', 'test')}
        for game_id in range(512):
            actor = trajectory_actor(config, game_id)
            self.assertEqual(actor, trajectory_actor(dict(config), game_id))
            counts[split_for_game(game_id)][actor] += 1
        for split, actors in counts.items():
            with self.subTest(split=split):
                self.assertGreater(actors['student'], 0)
                self.assertGreater(actors['teacher'], 0)
                self.assertTrue(.2 < actors['student']/sum(actors.values()) < .8)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy unavailable')
class RefinementAssemblyTests(unittest.TestCase):
    def fixture(self, root, name, offset, model_sha):
        import numpy as np
        from shogi_auto.policy.common import sha256, write_json
        directory=root/name;directory.mkdir()
        x=np.zeros((5,43,9,9),dtype=np.float32)
        for i in range(5):x[i,28].fill((offset+i+1)/18)
        legal=np.zeros((5,2187),dtype=np.bool_);legal[:,0]=True
        np.savez_compressed(directory/'data.npz',x=x,legal=legal,policy=np.zeros(5,dtype=np.int64),
            value=np.zeros(5,dtype=np.float32),split=np.array(['train']*3+['validation','test']),game_id=np.arange(5))
        write_json(directory/'data-manifest.json',{'source':{'commit':'fixture'},
            'npz_sha256':sha256(directory/'data.npz'),'counts':{'train':3,'validation':1,'test':1},
            'trajectory_model_sha256':model_sha})
        return directory,x

    def test_replay_keeps_only_train_and_excludes_diagnostic_states(self):
        import numpy as np
        from shogi_auto.policy.common import sha256
        from shogi_auto.policy.refinement import assemble
        from shogi_auto.policy.train import read_data
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'own.json';model.write_text('{}')
            fresh,_=self.fixture(root,'fresh',5,sha256(model))
            prior,x=self.fixture(root,'prior',0,sha256(model))
            excluded=root/'excluded.npz';np.savez_compressed(excluded,x=x[:1])
            with patch('shogi_auto.policy.refinement.source_identity',return_value={'dirty':False}), \
                    patch('shogi_auto.policy.refinement.load_network',return_value=object()), \
                    patch('shogi_auto.policy.refinement.weights_digest',return_value='fixture'):
                report=assemble(fresh,prior,model,root/'combined',exclude_data=[excluded])
            arrays,_=read_data(root/'combined')
            self.assertEqual(report['replayed_train_rows'],2)
            self.assertEqual(report['excluded_replay_train_rows'],1)
            self.assertEqual(report['replayed_validation_rows'],0)
            self.assertEqual(report['replayed_test_rows'],0)
            self.assertEqual(report['counts'],{'train':5,'validation':1,'test':1})
            self.assertTrue(set(arrays['game_id'][5:]).isdisjoint(set(arrays['game_id'][:5])))

    def test_fresh_diagnostic_overlap_is_rejected(self):
        import numpy as np
        from shogi_auto.policy.common import sha256
        from shogi_auto.policy.refinement import assemble
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'own.json';model.write_text('{}')
            fresh,x=self.fixture(root,'fresh',5,sha256(model))
            prior,_=self.fixture(root,'prior',0,sha256(model))
            excluded=root/'excluded.npz';np.savez_compressed(excluded,x=x[:1])
            with patch('shogi_auto.policy.refinement.source_identity',return_value={'dirty':False}):
                with self.assertRaisesRegex(ValueError,'overlaps diagnostics'):
                    assemble(fresh,prior,model,root/'combined',exclude_data=[excluded])


if __name__ == '__main__':
    unittest.main()
