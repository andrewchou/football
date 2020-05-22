import random
from collections import namedtuple

import numpy as np

class HistoryItem(namedtuple('HistoryItem', [
    'old_state', 'action', 'new_state', 'reward',
])):
    pass

class History():
    def __init__(self, max_size=100000):
        self.history = []
        self.index = 0
        self.max_size = max_size

    def add(self, item):
        assert isinstance(item, HistoryItem), item
        if len(self.history) < self.max_size:
            self.history.append(item)
        else:
            self.history[self.index] = item
            self.index = (self.index + 1) % self.max_size

    def sample(self, n=None):
        if n is None:
            all_history = self.history.copy()
            random.shuffle(all_history)
            return all_history
        return random.sample(population=self.history, k=min(n, len(self.history)))

    def mean_reward(self):
        return np.mean([item.reward for item in self.history])
