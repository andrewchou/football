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

"""Example football client.

It creates remote football game with given credentials and plays a few games.
"""

import random

import grpc
import numpy as np
import tensorflow.compat.v2 as tf
from absl import app
from absl import flags
from absl import logging

from gfootball.env.football_action_set import DEFAULT_ACTION_SET
from gfootball.env.football_env import FootballEnv

import gfootball.env as football_env
from gfootball.env import football_action_set
from gfootball.env.config import Config
from gfootball.env.players import agent_1v1, agent_rl_1v1
from gfootball.env.players.ppo2_cnn import Player

NUM_ACTIONS = len(football_action_set.ACTION_SET_DICT['default'])

def get_flags():
    FLAGS = flags.FLAGS
    flags.DEFINE_string('username', None, 'Username to use')
    flags.mark_flag_as_required('username')
    flags.DEFINE_string('token', None, 'Token to use.')
    flags.DEFINE_integer('how_many', 1000, 'How many games to play')
    flags.DEFINE_bool('render', False, 'Whether to render a game.')
    flags.DEFINE_string('track', '', 'Name of the competition track.')
    flags.DEFINE_string('model_name', '',
        'A model identifier to be displayed on the leaderboard.')
    # flags.DEFINE_string('inference_model', '',
    #     'A path to an inference model. Empty for random actions')
    flags.DEFINE_string('checkpoint', '',
        'A path to an checkpoint saved by run_ppo2.py. Empty for random actions')
    flags.DEFINE_enum('action_set', 'default', ['default', 'full'], 'Action set')
    return FLAGS

def random_actions(obs):
    num_players = 1 if len(obs.shape) == 3 else obs.shape[0]
    a = []
    for _ in range(num_players):
        a.append(random.randint(0, NUM_ACTIONS - 1))
    return a

def seed_rl_preprocessing(observation):
    observation = np.expand_dims(observation, axis=0)
    data = np.packbits(observation, axis=-1)  # This packs to uint8
    if data.shape[-1] % 2 == 1:
        data = np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, 1)], 'constant')
    return data.view(np.uint16)

def generate_actions(obs, model):
    a = []
    # Single agent case
    if len(obs.shape) == 3:
        a.append(model(seed_rl_preprocessing(obs))[0][0].numpy())
    else:
        # Multiagent -> first dimension is a number of agents you control.
        for x in range(obs.shape[0]):
            a.append(model(seed_rl_preprocessing(obs[x]))[0][0].numpy())
    return a

def get_inference_model(inference_model):
    if not inference_model:
        return random_actions
    model = tf.saved_model.load(inference_model)
    return lambda obs: generate_actions(obs, model)

def get_player(checkpoint):
    player_config = {
        'index': 0,
        'player_keyboard': 0,
        # 'player_ppo2_cnn': 0,
        'left_players': 1,
        'right_players': 0,
        'checkpoint': checkpoint,
    }
    env_config = {
        'action_set': 'default',
        'pitch_scale': 1.0,
        'warmstart': False,
        'random_frac': 0.0,
        'verbose': False,
    }
    # return Player(player_config=player_config, env_config=None)
    # return agent_1v1.Player(player_config=player_config, env_config=env_config)
    return agent_rl_1v1.Player(player_config=player_config, env_config=env_config)

def main(unused_argv):
    # model = get_inference_model(FLAGS.inference_model)
    level = {
        'mini': '1_vs_1_easy',
    }[FLAGS.track]
    # config = Config({
    #     'action_set': FLAGS.action_set,
    #     'dump_full_episodes': True,
    #     'players': [
    #         'keyboard:right_players=1',
    #         'ppo2_cnn:left_players=1,checkpoint=%s' % FLAGS.checkpoint,
    #     ],
    #     'real_time': False,
    #     'pitch_scale': 1.0,
    #     'level': level,
    # })
    player = get_player(checkpoint=FLAGS.checkpoint)
    env = football_env.create_remote_environment(
        FLAGS.username, FLAGS.token, FLAGS.model_name, track=FLAGS.track,
        # representation='extracted',
        representation='raw',
        stacked=False,
        include_rendering=FLAGS.render)
    for _ in range(FLAGS.how_many):
        obs = env.reset()
        cnt = 1
        done = False
        while not done:
            # try:
            # action = model(ob)
            print(obs)
            assert len(obs) == 1, len(obs)
            # adopted_obs = FootballEnv.convert_observations_static(
            #     original=obs[0], player=player,
            #     left_player_position=0,
            #     right_player_position=0,
            #     config=config,
            # )
            # a = self._action_to_list(player.take_action(adopted_obs))
            actions = player.take_action(observations=obs)
            assert len(actions) == 1, actions
            # while int(actions[0]._backend_action) >= NUM_ACTIONS:
            #     print(actions)
            #     actions = player.take_action(observations=obs)
            #     assert len(actions) == 1, actions
            print(actions)
            for k, v in sorted(obs[0].items()): print(k, v)
            ACTION_TO_INDEX_MAP = {a:i for i, a in enumerate(DEFAULT_ACTION_SET)}
            actions = [ACTION_TO_INDEX_MAP[a] for a in actions]
            obs, rew, done, _ = env.step(actions)
            logging.info('Playing the game, step %d, action %s, rew %s, done %d',
                cnt, actions, rew, done)
            cnt += 1
            # except grpc.RpcError as e:
            #     print(e)
            #     break
        print('=' * 50)

if __name__ == '__main__':
    FLAGS = get_flags()
    app.run(main)
