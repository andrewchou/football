import unittest
from collections import defaultdict
from unittest import TestCase

import numpy as np

from gfootball.common.state.angle import relative_angle_bucket
from gfootball.env.football_action_set import DEFAULT_ACTION_SET, action_right, action_bottom_right, ActionSetType
from gfootball.env.players.base_rl_agent import BaseRLPlayer
from gfootball.policies.base_policy import PolicyConfig, PolicyType

class BaseRLPlayerTestCase(TestCase):
    def get_player(self):
        # Mock
        return BaseRLPlayer(
            player_config={
                'left_players': 1,
                'right_players': 0,
                'warmstart': True,
                'verbose': False,
                'video': None,
                'policy_config': PolicyConfig(
                    policy_type=PolicyType.Q_LEARNING,
                    checkpoint=None,
                    random_frac=0.0,
                    action_set=DEFAULT_ACTION_SET,
                    lr=0,
                    discount=0.99,
                    n_steps=1,
                    verbose=False,
                ),
            },
            env_config={
                'action_set': ActionSetType.DEFAULT,
                'pitch_scale': 1.0,
            },
            testing=True
        )

    def test_avoid_opponent(self):
        player = self.get_player()
        action = player._avoid_opponent(
            own_position=np.array([0.0117, -1e-6]),
            opponent_position=np.array([0.06, -0.02]),
            target=np.array([0.85, 0]))
        self.assertEqual(action_bottom_right, action)
        action = player._avoid_opponent(
            own_position=np.array([0.022, 0.0012]),
            opponent_position=np.array([0.06, -0.02]),
            target=np.array([0.85, 0]))
        self.assertEqual(action_bottom_right, action)

    def test_direction_action(self):
        player = self.get_player()
        own_position = np.array([-0.06, -0.005])
        move_target = np.array([0.85, 0])
        action = player._direction_action(
            delta=move_target - own_position)
        self.assertEqual(action_right, action)


if __name__ == '__main__':
    unittest.main()
