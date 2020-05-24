from collections import namedtuple
from enum import Enum

class PolicyType(Enum):
    Q_LEARNING = 'Q_LEARNING'
    N_STEP_SARSA = 'N_STEP_SARSA'

class PolicyConfig(namedtuple('PolicyConfig', [
    'policy_type', 'checkpoint', 'random_frac', 'action_set',
    'lr', 'discount', 'n_steps',
    'verbose',
])):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        assert isinstance(self.policy_type, PolicyType), self
        return self

class BasePolicy(object):
    def __init__(self, policy_config):
        assert isinstance(policy_config, PolicyConfig), policy_config
        self.policy_config = policy_config
        self.load(checkpoint=policy_config.checkpoint)

    def load(self, checkpoint):
        assert 0, 'Need to implement load(checkpoint)'

    def save(self, checkpoint):
        assert 0, 'Need to implement save(checkpoint)'

    def process_epoch(self, items):
        assert 0, 'Need to implement process_epoch(items)'

    def get_action(self, state, debug=None):
        assert 0, 'Need to implement get_action(state, debug)'
