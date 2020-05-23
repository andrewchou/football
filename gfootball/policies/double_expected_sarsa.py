import pickle
import random
from collections import defaultdict

from gfootball.common.history import HistoryItem
from gfootball.env.football_action_set import DEFAULT_ACTION_SET

class DoubleExpectedSarsa():
    def __init__(
        self, random_frac=0.01, checkpoint=None, action_set=DEFAULT_ACTION_SET,
        verbose=False,
    ):
        self.random_frac = random_frac
        self.action_set = action_set
        self.Q1 = defaultdict(lambda: defaultdict(float))
        self.Q2 = defaultdict(lambda: defaultdict(float))
        if checkpoint:
            self.load(checkpoint=checkpoint)
        self.verbose = verbose

    def _load_single_Q(self, Q_json, Q):
        n = 0
        for state, action_values_dict in Q_json.items():
            for action, value in action_values_dict.items():
                if value != 0.0:
                    # print('LOADED:', state, action, value)
                    Q[state][action] = value
                    n += 1
        print('Loaded %d Q values.' % n)
        return Q

    def load(self, checkpoint):
        with open(checkpoint, 'rb') as inf:
            QS = pickle.load(inf)
        self.Q1 = self._load_single_Q(Q_json=QS['Q1'], Q=defaultdict(lambda: defaultdict(float)))
        self.Q2 = self._load_single_Q(Q_json=QS['Q2'], Q=defaultdict(lambda: defaultdict(float)))

    def save(self, checkpoint):
        Q1 = {k: {k2: v2 for k2, v2 in v.items() if v2 != 0.0} for k, v in self.Q1.items()}
        Q2 = {k: {k2: v2 for k2, v2 in v.items() if v2 != 0.0} for k, v in self.Q2.items()}
        n1 = sum([len(v) for v in Q1.values()])
        n2 = sum([len(v) for v in Q2.values()])
        with open(checkpoint, 'wb') as f:
            pickle.dump({
                'Q1': Q1,
                'Q2': Q2,
            }, f)
        print('Saved %d Q1 values, and %d Q2 values.' % (n1, n2))

    def give_reward(self, item):
        assert isinstance(item, HistoryItem), item
        # assert isinstance(old_state,)
        # old_state = self.get_state(observations=item.old_state)
        old_state = item.old_state
        new_state = item.new_state
        # Use Double Estimated SARSA
        # "Double" removes bias, and "Estimated" removes variance.
        if random.random() < 0.5:
            Q1 = self.Q1
            Q2 = self.Q2
        else:
            Q1 = self.Q2
            Q2 = self.Q1
        possible_actions_dict = Q1[new_state]
        best_action_value = max(possible_actions_dict.values()) if possible_actions_dict else 0
        probs_by_action = {
            action: self.random_frac / len(DEFAULT_ACTION_SET)
            for action in DEFAULT_ACTION_SET
        }
        list_of_best_actions = []
        for action in DEFAULT_ACTION_SET:
            value = possible_actions_dict[action]
            if value == best_action_value:
                list_of_best_actions.append(action)
        assert list_of_best_actions
        for action in list_of_best_actions:
            probs_by_action[action] += (1 - self.random_frac) / len(list_of_best_actions)
        expected_sarsa_state_value = 0.0
        for action, prob in probs_by_action.items():
            # Q2 is only used here to estimate the state value, after the state probs have been chosen by Q1
            expected_sarsa_state_value += prob * Q2[new_state][action]
        alpha = 1e-4
        discount = 0.999
        Q1[old_state][item.action] = (
            (1.0 - alpha) * Q1[old_state][item.action] +
            alpha * (item.reward + discount * expected_sarsa_state_value)
        )
        assert isinstance(Q1[old_state][item.action], float), Q1[old_state][item.action]

    def get_action(self, state, debug=None):
        possible_actions_dict1 = self.Q1[state]
        possible_actions_dict2 = self.Q2[state]
        possible_actions_dict = possible_actions_dict1.copy()
        for action, value in possible_actions_dict2.items():
            possible_actions_dict[action] += value
        if (not possible_actions_dict) or (random.random() < self.random_frac):
            action = self.action_set[random.randint(0, len(self.action_set) - 1)]
            if self.verbose and (debug is not None):
                debug.append(('Random action:', action))
        else:
            assert possible_actions_dict
            for a in self.action_set:
                if a not in possible_actions_dict:
                    possible_actions_dict[a] = 0.0
            best_value = max(possible_actions_dict.values())
            top_actions = [a for a in self.action_set if possible_actions_dict[a] == best_value]
            action = top_actions[random.randint(0, len(top_actions) - 1)]
            debug.append((' Value action:', action, best_value, min(possible_actions_dict.values())))
            assert isinstance(best_value, float)
        return action