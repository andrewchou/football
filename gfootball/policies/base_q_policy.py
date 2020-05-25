import random

import numpy as np

from gfootball.common.qdict import QDict
from gfootball.policies.base_policy import BasePolicy

class BaseQPolicy(BasePolicy):
    def __init__(self, policy_config):
        super().__init__(policy_config=policy_config)

    def _get_empty_Q(self):
        return QDict()

    def _load_single_Q(self, Q_json):
        Q = self._get_empty_Q()
        return Q

    def load(self, checkpoint):
        self.Q = self._get_empty_Q()
        if checkpoint:
            loaded = np.load(checkpoint, allow_pickle=True)
            Q_json = loaded['Q'].all()
            n = 0
            for state, action_values_dict in Q_json.items():
                for action, value in action_values_dict.items():
                    # print('LOADED:', state, action, value)
                    self.Q[state][action] = value
                    n += 1
            print('Loaded %d Q values.' % n)

    def save(self, checkpoint):
        Q = {k: {k2: v2 for k2, v2 in v.items() if v2 != 0.0} for k, v in self.Q.items()}
        np.savez_compressed(file=checkpoint, Q=Q)
        n = sum([len(v) for v in Q.values()])
        print('Saved %d Q values.' % n)

    def process_epoch(self, items):
        assert 0, 'Need to implement process_epoch(items)'

    def get_action(self, state, debug=None):
        if (state not in self.Q) or (random.random() < self.policy_config.random_frac):
            action = self.policy_config.action_set[random.randint(0, len(self.policy_config.action_set) - 1)]
            if self.policy_config.verbose and (debug is not None):
                debug.append(('Random action:', action))
        else:
            top_actions, best_value = self._get_best_actions_and_value(state=state)
            action = top_actions[random.randint(0, len(top_actions) - 1)]
            debug.append((' Value action:', action, best_value))
            assert isinstance(best_value, float)
        return action

    def _get_best_actions_and_value(self, state):
        possible_actions_dict = self.Q[state]
        best_value = safe_max(possible_actions_dict.values())
        top_actions = [a for a in self.policy_config.action_set if possible_actions_dict.get(a, 0.0) == best_value]
        return top_actions, best_value

def safe_min(l):
    if len(l) == 0:
        return 0.0
    return min(l)

def safe_max(l):
    if len(l) == 0:
        return 0.0
    return max(l)

