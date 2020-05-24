from gfootball.policies.base_q_policy import BaseQPolicy

class NStepSarsa(BaseQPolicy):
    def __init__(self, policy_config):
        super().__init__(policy_config=policy_config)

    def _get_q_value(self, items, t):
        return self.Q[items[t].old_state][items[t].action]

    def _set_q_value(self, items, t, value):
        self.Q[items[t].old_state][items[t].action] = value

    def process_epoch(self, items):
        # shorthand vars
        alpha = self.policy_config.lr
        discount = self.policy_config.discount
        n = self.policy_config.n_steps
        T = len(items)
        if self.policy_config.verbose: print('T', T)

        # t is the time being updated
        for t in range(T):
            if self.policy_config.verbose: print('t', t)

            # Calculate the discounted sampled reward.
            G = 0.0
            if self.policy_config.verbose: print('range', t + 1, min(t + n + 1, T) + 1)
            for i in range(t + 1, min(t + n + 1, T) + 1):
                reward = items[i - 1].reward
                if self.policy_config.verbose: print('i, reward', i, reward)
                G += (discount ** (i - t - 1)) * reward

            if self.policy_config.verbose: print('G1', G)
            if t + n < T:
                G += (discount ** n) * self._get_q_value(items=items, t=t + n)

            # Calculate the merged value
            if self.policy_config.verbose: print('G2', G)
            old_value = self._get_q_value(items=items, t=t)
            if self.policy_config.verbose: print('old_value', old_value)
            new_value = (1.0 - alpha) * old_value + alpha * G

            # Update the value
            if self.policy_config.verbose: print('new_value', new_value)
            self._set_q_value(items=items, t=t, value=new_value)
