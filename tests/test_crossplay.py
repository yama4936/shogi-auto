import unittest
from shogi_auto.policy.data import trajectory_actor,trajectory_student_color
from shogi_auto.learning.data import split_for_game

class CrossplayTests(unittest.TestCase):
    def test_crossplay_is_fixed_and_both_colors_present_in_every_split(self):
        c={'seed':2026110901,'seed_namespace':'yama-dr7-crossplay-v1-20261108','crossplay':True}
        seen={s:set() for s in ('train','validation','test')}
        for game in range(512):
            self.assertEqual(trajectory_actor(c,game),'crossplay')
            color=trajectory_student_color(c,game);self.assertEqual(color,trajectory_student_color(c,game));seen[split_for_game(game)].add(color)
        self.assertTrue(all(colors=={0,1} for colors in seen.values()))

    def test_old_mixed_actor_mode_is_preserved(self):
        c={'seed':12};self.assertEqual({trajectory_actor(c,i) for i in range(64)},{'student','teacher'})

if __name__=='__main__':unittest.main()
