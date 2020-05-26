"""Sample bot player."""
from collections import namedtuple

import numpy as np

from gfootball.env import football_action_set
from gfootball.env.players.base_rl_agent import BaseRLPlayer

class Player(BaseRLPlayer):
    def __init__(self, player_config, env_config):
        super().__init__(player_config=player_config, env_config=env_config)
        self._shoot_distance = 0.25

    def _get_hardcoded_action(self, debug):
        """Returns action to perform for the current observations."""
        active = self._get_own_position()
        # Corner etc. - just pass the ball
        if self._observation['game_mode'] != 0:
            if self.verbose:
                debug.append(('Not in the run of play. Mode:', self._observation['game_mode']))
            sticky_actions = self._get_sticky_actions()
            if not sticky_actions[4]:
                return football_action_set.action_right
            return football_action_set.action_long_pass

        #
        if self._they_have_the_ball():
            if self.verbose:
                debug.append('OTHER_TEAM_HAS_THE_BALL')
            move_target = self._get_ball_owner_location_target()
            return self._direction_action(move_target - active)
            # if self._last_action == football_action_set.action_pressure:
            #     return football_action_set.action_sprint
            # self._pressure_enabled = True
            # return football_action_set.action_pressure
        if not self._we_have_the_ball():
            move_target = self._get_ball_location()
            return self._direction_action(move_target - active)

        # if self._pressure_enabled:
        #     self._pressure_enabled = False
        #     return football_action_set.action_release_pressure
        target_x = 0.85 if self.pitch_scale == 1.0 else 0.35
        target_x = max(target_x, self._get_own_position()[0])

        GOOD_SPOT_TO_SHOOT_FROM = (target_x, 0)
        distance_from_good_spot_to_shoot_from = np.linalg.norm(
            self._get_ball_location() - GOOD_SPOT_TO_SHOOT_FROM)
        if self.verbose:
            debug.append((
                'distance_from_good_spot_to_shoot_from', distance_from_good_spot_to_shoot_from,
                self._get_ball_location(), GOOD_SPOT_TO_SHOOT_FROM))
        if distance_from_good_spot_to_shoot_from < self._shoot_distance:
            if self.verbose:
                debug.append('SHOOTING')
            return football_action_set.action_shot

        move_target = GOOD_SPOT_TO_SHOOT_FROM
        # Compute run direction.

        closest_front_opponent = self._closest_front_opponent(o=active, target=move_target)
        if closest_front_opponent is not None:
            dist_front_opp = self._object_distance(active, closest_front_opponent)
        else:
            dist_front_opp = 2.0

        # Maybe avoid opponent on your way?
        if dist_front_opp < 0.08:
            return self._avoid_opponent(
                own_position=active, opponent_position=closest_front_opponent, target=move_target)
        else:
            return self._direction_action(move_target - active)

    def get_state(self, observations):
        assert len(observations) == 1, len(observations)
        observation = observations[0]
        del observations
        if observation['ball_owned_player'] == -1:
            assert observation['ball_owned_team'] == -1, observation
        else:
            assert observation['ball_owned_team'] in (0, 1), observation
        field_position = []
        own_position = self._get_own_position()
        if own_position[0] < 0:
            field_position.append('B')
        else:
            field_position.append('F')
        if own_position[1] > 0.14 * self.pitch_scale:
            field_position.append('R')  # Right side, aka closer to the camera
        elif own_position[1] < -0.14 * self.pitch_scale:
            field_position.append('L')  # Left side, aka further to the camera
        else:
            field_position.append('C')
        assert observation['game_mode'] in set(range(7)), observation['game_mode']
        return BasicState(
            ball_owned_team=observation['ball_owned_team'],
            field_position=tuple(field_position),
            ball_angle_bucket=self._ball_angle_bucket_relative_to_me(),  # 6 (60 degree buckets)
            ball_close=bool(self._ball_distance_from_me() < 0.2),  # 2 (close, far)
            opponent_angle_bucket=self._opponent_angle_bucket_relative_to_me(),
            opponent_close=bool(self._opponent_distance_from_me() < 0.2),
            sticky_actions=tuple(observation['sticky_actions']),
            run_of_play=bool(observation['game_mode'] == 0),
        )

class BasicState(namedtuple('BasicState', [
    'ball_owned_team',  # 3
    'field_position',  # 6 spots (front/back, right/left/center)
    'ball_angle_bucket',  # 8 (45 degree buckets)
    'ball_close',  # 2 (close, far)
    'opponent_angle_bucket',  # 6 (60 degree buckets)
    'opponent_close',  # 2 (close, far)
    'sticky_actions',  # Many :(
    'run_of_play',  # 2 (normal vs all others)
])):
    def __new__(cls, *args, **kwargs):
        self = super(BasicState, cls).__new__(cls, *args, **kwargs)
        assert self.ball_owned_team in (-1, 0, 1), self
        assert isinstance(self.field_position, tuple), self
        assert len(self.field_position) == 2, self
        assert self.field_position[0] in ['F', 'B'], self
        assert self.field_position[1] in ['R', 'L', 'C'], self
        assert self.ball_angle_bucket in (0, 1, 2, 3, 4, 5, 6, 7), self
        assert isinstance(self.ball_close, bool), self
        assert self.opponent_angle_bucket in (0, 1, 2, 3, 4, 5, 6, 7), self
        assert isinstance(self.opponent_close, bool), self
        assert isinstance(self.sticky_actions, tuple), self
        assert len(self.sticky_actions) == 10, self
        # Example of more than 2: sticky_actions=(1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        # assert sum(self.sticky_actions) in [0, 1, 2], self
        assert isinstance(self.run_of_play, bool), self
        return self
