from collections import namedtuple

from gfootball.env import football_action_set
from gfootball.env.players.base_rl_agent import BaseRLPlayer

class Player(BaseRLPlayer):
    def __init__(self, player_config, env_config):
        super().__init__(player_config=player_config, env_config=env_config)
        self._shoot_distance = 0.25

    def _get_hardcoded_action(self, debug):
        self.n += 1
        if self.n <= 3:
            debug.append(('RIGHT', self.n))
            return football_action_set.action_right
        if self.n % 40 == 20:
            debug.append(('SPRINT', self.n))
            return football_action_set.action_sprint
        elif self.n % 40 == 0:
            debug.append(('STOP SPRINT', self.n))
            return football_action_set.action_release_sprint
        debug.append(('IDLE', self.n))
        return football_action_set.action_idle
        # return football_action_set.action_idle

    def get_state(self, observations):
        assert len(observations) == 1, len(observations)
        return BasicState()

    def reset(self):
        super().reset()
        self.n = 0

class BasicState(namedtuple('BasicState', [])):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        return self