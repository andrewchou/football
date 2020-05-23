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


"""Allows different types of players to play against each other."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import importlib
from os.path import expanduser

import gym
import numpy as np
from absl import logging

from gfootball.env.config import Config
from gfootball.env.players import keyboard, ppo2_cnn, agent_1v1, agent_rl_1v1, agent_rl_3v3

from gfootball.env import config as cfg
from gfootball.env import constants
from gfootball.env import football_action_set
from gfootball.env import football_env_core
from gfootball.env import observation_rotation

PLAYERS_BY_NAME = {
    'keyboard': keyboard.Player,
    'ppo2_cnn': ppo2_cnn.Player,
    'agent_1v1': agent_1v1.Player,
    'agent_rl_1v1': agent_rl_1v1.Player,
    'agent_rl_3v3': agent_rl_3v3.Player,
}
class FootballEnv(gym.Env):
    """Allows multiple players to play in the same environment."""

    def __init__(self, config, base_player_config=None):
        self._config = config
        player_config = {
            'index': 0,
        }
        if base_player_config:
            player_config.update(base_player_config)
        # There can be at most one agent at a time. We need to remember its
        # team and the index on the team to generate observations appropriately.
        self._agent = None
        self._agent_index = -1
        self._agent_left_position = -1
        self._agent_right_position = -1
        self._players = self._construct_players(definitions=config['players'], base_player_config=player_config)
        self._env = football_env_core.FootballEnvCore(self._config)
        self._num_actions = len(football_action_set.get_action_set(self._config))
        self._cached_observation = None

    @property
    def action_space(self):
        if self._config.number_of_players_agent_controls() > 1:
            return gym.spaces.MultiDiscrete(
                [self._num_actions] * self._config.number_of_players_agent_controls())
        return gym.spaces.Discrete(self._num_actions)

    def _construct_players(self, definitions, base_player_config):
        result = []
        left_position = 0
        right_position = 0
        for definition in definitions:
            (name, d) = cfg.parse_player_definition(definition)
            config_name = 'player_{}'.format(name)
            if config_name in base_player_config:
                base_player_config[config_name] += 1
            else:
                base_player_config[config_name] = 0
            if name in PLAYERS_BY_NAME:
                player_class = PLAYERS_BY_NAME[name]
            else:
                assert 0, 'Unknown player class: %s. Choose from: %s' % (name, sorted(PLAYERS_BY_NAME))
            player_config = copy.deepcopy(base_player_config)
            player_config.update(d)
            if ('checkpoint' in player_config) and (player_config['checkpoint'] is not None):
                player_config['checkpoint'] = expanduser(player_config['checkpoint'])
            player = player_class(player_config=player_config, env_config=self._config)
            if name.startswith('agent_'):
                assert not self._agent, 'Only one \'agent\' player allowed'
                self._agent = player
                self._agent_index = len(result)
                self._agent_left_position = left_position
                self._agent_right_position = right_position
            result.append(player)
            left_position += player.num_controlled_left_players()
            right_position += player.num_controlled_right_players()
            base_player_config['index'] += 1
        return result

    def _convert_observations(
        self, original, player, left_player_position, right_player_position,
    ):
        return FootballEnv.convert_observations_static(
            original=original, player=player,
            left_player_position=left_player_position,
            right_player_position=right_player_position,
            config=self._config,
        )

    @staticmethod
    def convert_observations_static(
        original, player, left_player_position, right_player_position, config,
    ):
        """Converts generic observations returned by the environment to
           the player specific observations.

        Args:
          original: original observations from the environment.
          player: player for which to generate observations.
          left_player_position: index into observation corresponding to the left
              player.
          right_player_position: index into observation corresponding to the right
              player.
        """
        assert isinstance(config, Config), config
        observations = []
        # assert 0, (original.keys(), player, left_player_position, right_player_position)
        for is_left in [True, False]:
            adopted = original if is_left or player.can_play_right(
            ) else observation_rotation.flip_observation(original, config)
            prefix = 'left' if is_left or not player.can_play_right() else 'right'
            position = left_player_position if is_left else right_player_position
            for x in range(player.num_controlled_left_players() if is_left
            else player.num_controlled_right_players()):
                o = {}
                for v in constants.EXPOSED_OBSERVATIONS:
                    # Active and sticky_actions are added below.
                    if v != 'active' and v != 'sticky_actions':
                        o[v] = copy.deepcopy(adopted[v])
                assert (len(adopted[prefix + '_agent_controlled_player']) == len(
                    adopted[prefix + '_agent_sticky_actions']))
                if position + x >= len(adopted[prefix + '_agent_controlled_player']):
                    o['active'] = -1
                    o['sticky_actions'] = []
                else:
                    o['active'] = (
                        adopted[prefix + '_agent_controlled_player'][position + x])
                    o['sticky_actions'] = np.array(copy.deepcopy(
                        adopted[prefix + '_agent_sticky_actions'][position + x]))
                # There is no frame for players on the right ATM.
                if is_left and 'frame' in original:
                    o['frame'] = original['frame']
                observations.append(o)
        return observations

    def _action_to_list(self, a):
        if isinstance(a, np.ndarray):
            return a.tolist()
        if not isinstance(a, list):
            return [a]
        return a

    def get_players_and_relative_obs_pairs(self, obs):
        players_and_relative_obs_pairs = []
        left_player_position = 0
        right_player_position = 0
        agent_obs = None
        for player_index, player in enumerate(self._players):
            adopted_obs = self._convert_observations(
                original=obs, player=player,
                left_player_position=left_player_position,
                right_player_position=right_player_position)
            left_player_position += player.num_controlled_left_players()
            right_player_position += player.num_controlled_right_players()
            players_and_relative_obs_pairs.append((player, adopted_obs))
            if player_index == self._agent_index:
                agent_obs = adopted_obs
        return players_and_relative_obs_pairs, agent_obs

    def _get_actions(self):
        obs = self._env.observation()
        players_and_relative_obs_pairs, agent_obs = self.get_players_and_relative_obs_pairs(obs=obs)
        # assert 0, obs
        left_actions = []
        right_actions = []
        for player, adopted_obs in players_and_relative_obs_pairs:
            a = self._action_to_list(player.take_action(adopted_obs))
            assert len(adopted_obs) == len(
                a), 'Player provided {} actions instead of {}.'.format(
                len(a), len(adopted_obs))
            if not player.can_play_right():
                for x in range(player.num_controlled_right_players()):
                    index = x + player.num_controlled_left_players()
                    a[index] = observation_rotation.flip_single_action(
                        a[index], self._config)
            left_actions.extend(a[:player.num_controlled_left_players()])
            right_actions.extend(a[player.num_controlled_left_players():])
        actions = left_actions + right_actions
        return actions, agent_obs

    def step(self):
        # action = self._action_to_list(action)
        # if self._agent:
        #     self._agent.set_action(action)
        # else:
        #     assert len(
        #         action
        #     ) == 0, 'step() received {} actions, but no agent is playing.'.format(
        #         len(action))

        actions, agent_obs = self._get_actions()
        _, reward, done, info = self._env.step(actions)
        score_reward = reward
        if self._agent:
            reward = ([reward] * self._agent.num_controlled_left_players() +
                      [-reward] * self._agent.num_controlled_right_players())
        # for player in self._players:
        #     if player.num_controlled_left_players() > 0:
        #         assert player.num_controlled_right_players() == 0
        #         player.give_reward(reward=reward)
        #     elif player.num_controlled_right_players() > 0:
        #         assert player.num_controlled_left_players() == 0
        #         player.give_reward(reward=-reward)
        self._cached_observation = None
        info['score_reward'] = score_reward
        if self._agent is not None:
            info['agent_action'] = actions[self._agent_index]
            info['agent_obs'] = actions[self._agent_index]
        return (self.observation(), np.array(reward, dtype=np.float32), done, info)

    def reset(self):
        self._env.reset()
        for player in self._players:
            player.reset()
        self._cached_observation = None
        return self.observation()

    def observation(self):
        if not self._cached_observation:
            self._cached_observation = self._env.observation()
            if self._agent:
                self._cached_observation = self._convert_observations(
                    self._cached_observation, self._agent,
                    self._agent_left_position, self._agent_right_position)
        return self._cached_observation

    def write_dump(self, name):
        return self._env.write_dump(name)

    def close(self):
        self._env.close()

    def get_state(self, to_pickle={}):
        return self._env.get_state(to_pickle)

    def set_state(self, state):
        self._cached_observation = None
        return self._env.set_state(state)

    def tracker_setup(self, start, end):
        self._env.tracker_setup(start, end)

    def render(self, mode='human'):
        self._cached_observation = None
        return self._env.render(mode=mode)

    def disable_render(self):
        self._cached_observation = None
        return self._env.disable_render()
