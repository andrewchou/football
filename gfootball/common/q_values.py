from collections import defaultdict

class Q(object):
    def __init__(self):
        self.q_sums = defaultdict(lambda: defaultdict(float))
        self.q_cnts = defaultdict(lambda: defaultdict(float))
        self.default_value = 0.0

    def add(self, state, action, reward):
        assert isinstance(state, tuple), state
        assert isinstance(action, int), action
        assert isinstance(reward, float), reward
        self.q_cnts[state][action] += 1
        self.q_sums[state][action] += reward

    def get_q_value(self, state, action):
        cnt = self.q_cnts[state][action]
        if cnt == 0:
            return self.default_value
        return self.q_sums[state][action] / cnt

    def get_v_value(self, state):
        possible_actions_dict = self.q_cnts[state]
        return max(possible_actions_dict.values()) if possible_actions_dict else self.default_value
