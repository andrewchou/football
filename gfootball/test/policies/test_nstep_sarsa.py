import unittest
from unittest import TestCase

from gfootball.common.history import HistoryItem
from gfootball.env.football_action_set import DEFAULT_ACTION_SET, CoreAction, action_shot, action_short_pass, action_long_pass, action_idle
from gfootball.policies.base_policy import PolicyType, PolicyConfig
from gfootball.policies.nstep_sarsa import NStepSarsa

class NStepSarsaTestCase(TestCase):
    def get_policy(self, n_steps=5):
        return NStepSarsa(
            policy_config=PolicyConfig(
                policy_type=PolicyType.N_STEP_SARSA,
                checkpoint=None,
                random_frac=0.0,
                action_set=DEFAULT_ACTION_SET,
                lr=0.25,
                discount=0.9,
                n_steps=n_steps,
                verbose=False,
            ),
        )

    def test_single_item(self):
        policy = self.get_policy()
        policy.process_epoch(items=[
            HistoryItem(
                old_state=tuple(['A']),
                action=action_shot,
                new_state=tuple(['B']),
                reward=1.0,
            ),
        ])
        self.assertEqual({('A',): {action_shot: 0.25}} , policy.Q)

    def test_a_few_items(self):
        policy = self.get_policy()
        policy.process_epoch(items=[
            HistoryItem(
                old_state=tuple(['A']),
                action=action_short_pass,
                new_state=tuple(['B']),
                reward=0.0,
            ),
            HistoryItem(
                old_state=tuple(['B']),
                action=action_long_pass,
                new_state=tuple(['C']),
                reward=0.0,
            ),
            HistoryItem(
                old_state=tuple(['C']),
                action=action_idle,
                new_state=tuple(['D']),
                reward=0.0,
            ),
            HistoryItem(
                old_state=tuple(['D']),
                action=action_shot,
                new_state=tuple(['E']),
                reward=1.0,
            ),
        ])
        self.assertAlmostEqual({
            ('A',): {action_short_pass: 0.18225000000000002},  # TODO round this
            ('B',): {action_long_pass: 0.2025},
            ('C',): {action_idle: 0.225},
            ('D',): {action_shot: 0.25},
        }, policy.Q)

    def test_existing_value_for_state_being_updated(self):
        policy = self.get_policy()
        policy.Q[('A',)][action_shot] = 0.2
        policy.process_epoch(items=[
            HistoryItem(
                old_state=tuple(['A']),
                action=action_shot,
                new_state=tuple(['B']),
                reward=1.0,
            ),
        ])
        self.assertEqual({('A',): {action_shot: 0.4}} , policy.Q)

    def test_existing_value_for_horizon_state(self):
        policy = self.get_policy(n_steps=2)
        policy.Q[('C',)][action_idle] = 0.25
        policy.process_epoch(items=[
            HistoryItem(
                old_state=tuple(['A']),
                action=action_short_pass,
                new_state=tuple(['B']),
                reward=0.0,
            ),
            HistoryItem(
                old_state=tuple(['B']),
                action=action_long_pass,
                new_state=tuple(['C']),
                reward=0.0,
            ),
            HistoryItem(
                old_state=tuple(['C']),
                action=action_idle,
                new_state=tuple(['D']),
                reward=0.0,
            ),
        ])
        self.assertEqual({
            ('A',): {action_short_pass: 0.050625},
            ('B',): {action_long_pass: 0.0},
            ('C',): {action_idle: 0.1875},
        } , policy.Q)

if __name__ == '__main__':
    unittest.main()
