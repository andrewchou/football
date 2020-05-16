# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Sample bot player."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pygame

from gfootball.env import football_action_set
from gfootball.env import player_base
from gfootball.env.football_action_set import ACTION_SET_DICT

class Player(player_base.PlayerBase):

    def __init__(self, player_config, env_config):
        assert env_config["action_set"] == 'default'
        self.pitch_scale = env_config['pitch_scale']
        assert self.pitch_scale in (1.0, 0.5), self.pitch_scale
        player_base.PlayerBase.__init__(self, player_config)
        self._observation = None
        self._last_action = football_action_set.action_idle
        self._shoot_distance = 0.25  # 0.15
        self._pressure_enabled = False
        pygame.init()
        self._init_done = False

    def _object_distance(self, object1, object2):
        """Computes distance between two objects."""
        return np.linalg.norm(np.array(object1) - np.array(object2))

    def _direction_action(self, delta):
        """For required movement direction vector returns appropriate action."""
        all_directions = [
            football_action_set.action_top,
            football_action_set.action_top_left,
            football_action_set.action_left,
            football_action_set.action_bottom_left,
            football_action_set.action_bottom,
            football_action_set.action_bottom_right,
            football_action_set.action_right,
            football_action_set.action_top_right
        ]
        all_directions_vec = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1)]
        all_directions_vec = [
            np.array(v) / np.linalg.norm(np.array(v)) for v in all_directions_vec
        ]
        best_direction = np.argmax([np.dot(delta, v) for v in all_directions_vec])
        action = all_directions[best_direction]
        # TODO
        # if (self._last_action == action): # or (self._last_action == football_action_set.action_sprint):
        #     return football_action_set.action_idle
        return action

    def _closest_opponent_to_object(self, o):
        """For a given object returns the closest opponent.

        Args:
          o: Source object.

        Returns:
          Closest opponent."""
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            d = self._object_distance(o, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p
        assert closest is not None
        return closest

    def _closest_front_opponent(self, o, target):
        """For an object and its movement direction returns the closest opponent.

        Args:
          o: Source object.
          target: Movement direction.

        Returns:
          Closest front opponent."""
        delta = target - o
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            delta_opp = p - o
            if np.dot(delta, delta_opp) <= 0:
                continue
            d = self._object_distance(o, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p

        # May return None!
        return closest

    def _score_pass_target(self, active, player):
        """Computes score of the pass between players.

        Args:
          active: Player doing the pass.
          player: Player receiving the pass.

        Returns:
          Score of the pass.
        """
        opponent = self._closest_opponent_to_object(player)
        dist = self._object_distance(player, opponent)
        trajectory = player - active
        dist_closest_traj = None
        for i in range(10):
            position = active + (i + 1) / 10.0 * trajectory
            opp_traj = self._closest_opponent_to_object(position)
            dist_traj = self._object_distance(position, opp_traj)
            if dist_closest_traj is None or dist_traj < dist_closest_traj:
                dist_closest_traj = dist_traj
        return -dist_closest_traj

    def _best_pass_target(self, active):
        """Computes best pass a given player can do.

        Args:
          active: Player doing the pass.

        Returns:
          Best target player receiving the pass.
        """
        best_score = None
        best_target = None
        for player in self._observation['left_team']:
            if self._object_distance(player, active) > 0.3:
                continue
            score = self._score_pass_target(active, player)
            if best_score is None or score > best_score:
                best_score = score
                best_target = player
        return best_target

    def _avoid_opponent(self, active, opponent, target):
        """Computes movement action to avoid a given opponent.

        Args:
          active: Active player.
          opponent: Opponent to be avoided.
          target: Original movement direction of the active player.

        Returns:
          Action to perform to avoid the opponent.
        """
        # Choose a perpendicular direction to the opponent, towards the target.
        delta = opponent - active
        delta_t = target - active
        new_delta = [delta[1], -delta[0]]
        if delta_t[0] * new_delta[0] < 0:
            new_delta = [-new_delta[0], -new_delta[1]]

        return self._direction_action(new_delta)

    def _get_ball_location_target(self):
        return self._observation['ball'][:2]

    def _get_ball_owner_location_target(self):
        assert self._observation['ball_owned_team'] == 1, self._observation
        '''
        active 0
        ball [-0.6474961   0.06609182  0.31785655]
        ball_direction [-0.05493543 -0.0094937   0.30812848]
        ball_owned_player -1
        ball_owned_team -1
        ball_rotation [-2.30635778e-05 -2.16302942e-05 -2.27212766e-03]
        game_mode 0
        left_team [[-0.0212006 -0.290353 ]]
        left_team_active [ True]
        left_team_direction [[ 0.       -0.005981]]
        left_team_roles [0]
        left_team_tired_factor [0.05124319]
        left_team_yellow_card [False]
        right_team [[-0.6075235   0.07153132]]
        right_team_active [ True]
        right_team_direction [[-0.00908164  0.00055133]]
        right_team_roles [0]
        right_team_tired_factor [0.05855602]
        right_team_yellow_card [False]
        score [0, 0]
        steps_left 371
        sticky_actions [0 0 1 0 0 0 0 0 0 0]
        '''
        other_team_positions = self._observation['right_team']
        other_player_with_ball_position = other_team_positions[self._observation['ball_owned_player']]
        return other_player_with_ball_position - self._get_own_position()

    def _get_own_position(self):
        return self._observation['left_team'][self._observation['active']]

    def _they_have_the_ball(self):
        return self._observation['ball_owned_team'] == 1

    def _we_have_the_ball(self):
        return self._observation['ball_owned_team'] == 0

    def _get_sticky_actions(self):
        return self._observation['sticky_actions']

    def _get_action(self):
        """Returns action to perform for the current observations."""
        active = self._get_own_position()
        # Corner etc. - just pass the ball
        if self._observation['game_mode'] != 0:
            print('Not in the run of play. Mode:', self._observation['game_mode'])
            # print(self._observation)
            sticky_actions = self._get_sticky_actions()
            if not sticky_actions[4]:
                return football_action_set.action_right
            return football_action_set.action_long_pass

        #
        if self._they_have_the_ball():
            print('OTHER_TEAM_HAS_THE_BALL')
            move_target = self._get_ball_owner_location_target()
            return self._direction_action(move_target - active)
            # if self._last_action == football_action_set.action_pressure:
            #     return football_action_set.action_sprint
            # self._pressure_enabled = True
            # return football_action_set.action_pressure
        if not self._we_have_the_ball():
            move_target = self._get_ball_location_target()
            return self._direction_action(move_target - active)

        # if self._pressure_enabled:
        #     self._pressure_enabled = False
        #     return football_action_set.action_release_pressure
        target_x = 0.85 if self.pitch_scale == 1.0 else 0.35
        target_x = max(target_x, self._get_own_position()[0])

        GOOD_SPOT_TO_SHOOT_FROM = (target_x, 0)
        distance_from_good_spot_to_shoot_from = np.linalg.norm(
            self._get_ball_location_target() - GOOD_SPOT_TO_SHOOT_FROM)
        print('distance_from_good_spot_to_shoot_from', distance_from_good_spot_to_shoot_from, self._get_ball_location_target(), GOOD_SPOT_TO_SHOOT_FROM)
        if distance_from_good_spot_to_shoot_from < self._shoot_distance:
            print('SHOOTING')
            return football_action_set.action_shot

        move_target = GOOD_SPOT_TO_SHOOT_FROM
        # Compute run direction.
        move_action = self._direction_action(move_target - active)

        closest_front_opponent = self._closest_front_opponent(active, move_target)
        if closest_front_opponent is not None:
            dist_front_opp = self._object_distance(active, closest_front_opponent)
        else:
            dist_front_opp = 2.0

        # Maybe avoid opponent on your way?
        if dist_front_opp < 0.08:
            best_pass_target = self._best_pass_target(active)
            if np.array_equal(best_pass_target, active):
                move_action = self._avoid_opponent(active, closest_front_opponent,
                    move_target)
            else:
                delta = best_pass_target - active
                direction_action = self._direction_action(delta)
                if self._last_action == direction_action:
                    return football_action_set.action_short_pass
                else:
                    return direction_action
        return move_action

    def take_action(self, observations):
        if not self._init_done:
            self._init_done = True
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
        assert len(observations) == 1, 'Bot does not support multiple player control'
        print()
        print('OBS')
        for k, v in sorted(observations[0].items()):
            if k != 'frame':
                print(k, v)
        print()
        self._observation = observations[0]
        self._last_action = self._get_action()
        assert self._last_action in ACTION_SET_DICT['default'], self._last_action
        print(self._last_action)
        return [self._last_action]
