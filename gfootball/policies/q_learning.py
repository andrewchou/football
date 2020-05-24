from gfootball.common.history import HistoryItem
from gfootball.policies.base_q_policy import BaseQPolicy

class QLearning(BaseQPolicy):
    def __init__(self, policy_config):
        super().__init__(policy_config=policy_config)

    def _give_reward(self, item):
        assert isinstance(item, HistoryItem), item
        new_state = item.new_state
        top_actions, best_value = self._get_best_actions_and_value(state=new_state)
        alpha = self.policy_config.lr
        discount = 0.999
        self.Q[item.old_state][item.action] = (
            (1.0 - alpha) * self.Q[item.old_state][item.action] +
            alpha * (item.reward + discount * best_value)
        )

    def process_epoch(self, items):
        # reverse so that reward can propogate backward faster
        for item in reversed(items):
            self._give_reward(item=item)
