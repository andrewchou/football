'''Sample bot player.'''
import random
from collections import namedtuple

import numpy as np

from gfootball.env import football_action_set
from gfootball.env.football_action_set import DEFAULT_ACTION_SET
from gfootball.env.players.base_rl_agent import BaseRLPlayer
from gfootball.policies.double_expected_sarsa import DoubleExpectedSarsa
from gfootball.scenarios import e_PlayerRole_CB, e_PlayerRole_CF, e_PlayerRole_GK
from third_party import gfootball_engine

class Player(BaseRLPlayer):
    def __init__(self, player_config, env_config):
        super().__init__(player_config=player_config, env_config=env_config)
        self._shoot_distance = 0.15

    def _get_assigned_opponent_to_defend_target(self):
        assert self._observation['ball_owned_team'] == 1, self._observation
        other_team_positions = self._observation['right_team']
        ball_owner_index = self._observation['ball_owned_player']
        # 0 is the keeper
        n_players_per_team = len(other_team_positions)
        # TODO handle matchups for non-3v3
        if (ball_owner_index == 0) or (len(other_team_positions) != 3):
            # Match up with the opposite type of player
            other_player_with_ball_position = other_team_positions[n_players_per_team - self._get_own_index()]
        else:
            # TODO handle matchups for non-3v3
            assert ball_owner_index in (1, 2), ball_owner_index
            other_player_with_ball_position = other_team_positions[n_players_per_team - ball_owner_index]
        return other_player_with_ball_position - self._get_own_position()

    def _get_hardcoded_setpiece_action(self, debug):
        game_mode = self._observation['game_mode']
        debug.append(('Not in the run of play. Mode:', game_mode))
        if game_mode in (
            gfootball_engine.e_GameMode.e_GameMode_KickOff,
            gfootball_engine.e_GameMode.e_GameMode_Corner,
            gfootball_engine.e_GameMode.e_GameMode_FreeKick,
            gfootball_engine.e_GameMode.e_GameMode_ThrowIn,
        ):
            pass_player_index = self._closest_team_member_index_to_me()
            return self._get_hardcoded_pass_action(
                # TODO could be a long pass
                pass_player_index=pass_player_index, long_pass=False, debug=debug)
        if game_mode == gfootball_engine.e_GameMode.e_GameMode_Penalty:
            return self._get_hardcoded_shot_action(debug=debug)
        # if game_mode == gfootball_engine.e_GameMode.e_GameMode_GoalKick:
            # Otherwise accidently score on the keeper a lot hitting passing
            # return football_action_set.action_idle
        # There's a keeper, so dont worry about own goals
        # TODO face toward teammate and then pass
        return DEFAULT_ACTION_SET[random.randint(0, len(DEFAULT_ACTION_SET) - 1)]
        # return football_action_set.action_short_pass

    def _get_hardcoded_defensive_action(self, debug):
        debug.append('OTHER_TEAM_HAS_THE_BALL')
        if self._am_closest_team_member_to_opponent_with_ball() and (not self._their_gk_has_the_ball()):
            ball_owner_location = self._get_ball_owner_location()
            move_target = ball_owner_location - self._get_own_position()
            move_action = self._direction_action(delta=move_target)
            debug.append(('IM THE CLOSEST DEFENDER', move_target, move_action))
        else:
            move_target = self._get_assigned_opponent_to_defend_target()
            move_action = self._direction_action(move_target)
            debug.append(('IM SECONDARY DEFENDER', move_target, move_action))
        return move_action
        # if self._last_action == football_action_set.action_pressure:
        #     return football_action_set.action_sprint
        # self._pressure_enabled = True
        # return football_action_set.action_pressure

    def _get_hardcoded_loose_ball_action(self, debug):
        move_target = self._get_ball_location() - self._get_own_position()
        move_action = self._direction_action(move_target)
        debug.append(('RUNNING TO THE BALL:', move_action))
        return move_action

    def _get_hardcoded_shot_action(self, debug):
        # If you press pass multiple times in a row then the next player will just end up passing.
        if (football_action_set.action_shot in self._action_history[-4:]):
            move_target = self._opponent_goal_location() - self._get_own_position()
            debug.append(('POST SHOT DIRECTION', move_target))
            return self._direction_action(delta=move_target)
        debug.append(('SHOOTING', self._get_ball_location()))
        return football_action_set.action_shot

    def _get_hardcoded_pass_action(self, pass_player_index, long_pass, debug):
        assert pass_player_index not in (self._get_own_keeper_index(), self._get_own_index())
        delta = self._get_teammate_position(teammate_index=pass_player_index) - self._get_own_position()
        direction_action = self._direction_action(delta)
        if (
            self._action_history and
            (self._action_history[-1] == direction_action) and
            # If you press pass multiple times in a row then the next player will just end up passing.
            # TODO have the pass target run toward the ball
            (football_action_set.action_short_pass not in self._action_history[-4:]) and
            (football_action_set.action_long_pass not in self._action_history[-4:])
        ):
            debug.append('PASSING')
            if long_pass:
                return football_action_set.action_long_pass
            else:
                return football_action_set.action_short_pass
        else:
            debug.append(('PREPARING FOR PASS:', direction_action))
            return direction_action

    def _get_hardcoded_action(self, debug):
        '''Returns action to perform for the current observations.'''
        own_position = self._get_own_position()
        # Corner etc. - just pass the ball
        game_mode = self._observation['game_mode']
        if game_mode != gfootball_engine.e_GameMode.e_GameMode_Normal:
            return self._get_hardcoded_setpiece_action(debug=debug)
        if self._they_have_the_ball():
            return self._get_hardcoded_defensive_action(debug=debug)
        if not self._we_have_the_ball():
            return self._get_hardcoded_loose_ball_action(debug=debug)

        # if self._pressure_enabled:
        #     self._pressure_enabled = False
        #     return football_action_set.action_release_pressure
        target_x = self.pitch_scale - 0.15
        target_x = max(target_x, own_position[0])

        ball_location = self._get_ball_location()
        debug.append(('ball_location', ball_location))

        GOOD_SPOT_TO_SHOOT_FROM = (target_x, 0)
        if (
            (ball_location[0] > self.pitch_scale - 0.25) and
            (np.linalg.norm(ball_location - GOOD_SPOT_TO_SHOOT_FROM) < self._shoot_distance)
        ):
            return self._get_hardcoded_shot_action(debug=debug)

        move_target = GOOD_SPOT_TO_SHOOT_FROM

        closest_front_opponent = self._closest_front_opponent(own_position, move_target)
        if closest_front_opponent is not None:
            dist_front_opp = self._object_distance(own_position, closest_front_opponent)
        else:
            dist_front_opp = 2.0

        # Maybe avoid opponent on your way?
        if dist_front_opp < 0.08:
            best_pass_player_index, best_pass_player_position, best_is_long_pass = self._best_pass_player_index(debug=debug)
            # Dont pass to self or GK
            if (
                (best_pass_player_index in (self._get_own_index(), self._get_own_keeper_index())) or
                self._any_teammate_offside() # Only relevant in 3v3, need to check index if have more players
            ):
                move_action = self._avoid_opponent(
                    own_position=own_position, opponent_position=closest_front_opponent, target=move_target)
                debug.append(('DRIBBLING:', move_action))
            else:
                return self._get_hardcoded_pass_action(
                    pass_player_index=best_pass_player_index, long_pass=best_is_long_pass, debug=debug)
        else:
            # Compute run direction.
            move_action = self._direction_action(delta=move_target - own_position)
        return move_action

    def _get_teammate_position(self, teammate_index):
        active_index = self._get_own_index()
        assert teammate_index != active_index, (teammate_index, active_index)
        assert self._observation['left_team_roles'][active_index] != e_PlayerRole_GK
        return self._observation['left_team'][teammate_index]

    def _any_teammate_offside(self):
        active_index = self._get_own_index()
        for teammate_index, teammate_position in enumerate(self._observation['left_team']):
            if teammate_index == active_index:
                continue
            num_opponents_closer_to_their_goal = 0
            for opponent_pos in self._observation['right_team']:
                if teammate_position[0] < opponent_pos[0]:
                    num_opponents_closer_to_their_goal += 1
            offside = bool(
                (num_opponents_closer_to_their_goal < 2) and (teammate_position[0] > 0)
            )
            if offside:
                return True
        return False

    def get_state(self, observations):
        assert len(observations) == 1, len(observations)
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
        observation = observations[0]
        del observations
        if observation['ball_owned_player'] == -1:
            assert observation['ball_owned_team'] == -1, observation
        else:
            assert observation['ball_owned_team'] in (0, 1), observation
        field_position = []
        own_position = self._get_own_position()
        if own_position[0] < 0:
            # In defensive final third
            if own_position[0] < -self.pitch_scale + 0.25:
                field_position.append('B')
            else:
                field_position.append('BM')
        else:
            assert own_position[0] >= 0, own_position
            # In the final third
            if own_position[0] > self.pitch_scale - 0.25:
                field_position.append('F')
            else:
                field_position.append('FM')
        if own_position[1] > 0.14 * self.pitch_scale:
            field_position.append('R')  # Right side, aka closer to the camera
        elif own_position[1] < -0.14 * self.pitch_scale:
            field_position.append('L')  # Left side, aka further to the camera
        else:
            field_position.append('C')
        assert observation['game_mode'] in set(range(7)), observation['game_mode']
        closest_team_member_index_to_ball = self._closest_team_member_index_to_object(
            o=self._get_ball_location())
        return BasicState(
            ball_owned_team=observation['ball_owned_team'],
            field_position=tuple(field_position),
            ball_angle_bucket=self._ball_angle_bucket_relative_to_me(),  # 6 (60 degree buckets)
            ball_close=bool(self._ball_distance_from_me() < 0.1),
            closest_to_ball_on_my_team=bool(closest_team_member_index_to_ball == observation['active']),
            opponent_angle_bucket=self._opponent_angle_bucket_relative_to_me(),
            opponent_close=bool(self._opponent_distance_from_me() < 0.1),
            sticky_actions=tuple(observation['sticky_actions']),
            run_of_play=bool(observation['game_mode'] == 0),
            am_offside=self._am_offside(),
            teammate_offside=self._any_teammate_offside(),
            role=self._get_role(),
        )

class BasicState(namedtuple('BasicState', [
    'ball_owned_team',  # 3
    'field_position',  # 12 spots (front/front-middle/back-middle/back, right/left/center)
    'ball_angle_bucket',  # 8 (45 degree buckets)
    'ball_close',  # 2 (close, far)
    'closest_to_ball_on_my_team',  # 2 (yes, no)
    'opponent_angle_bucket',  # 6 (60 degree buckets)
    'opponent_close',  # 2 (close, far)
    # At most one of the first 8 can be on since they are directions.
    # The last two can be on independently
    'sticky_actions',
    'run_of_play',  # 2 (normal vs all others)
    'am_offside',  # 2 (yes, no)
    'teammate_offside',  # 2 (yes, no)
    'role',  # 2 (yes, no)
])):
    def __new__(cls, *args, **kwargs):
        self = super(BasicState, cls).__new__(cls, *args, **kwargs)
        assert self.ball_owned_team in (-1, 0, 1), self
        assert isinstance(self.field_position, tuple), self
        assert len(self.field_position) == 2, self
        assert self.field_position[0] in ['F', 'FM', 'BM', 'B'], self
        assert self.field_position[1] in ['R', 'L', 'C'], self
        assert self.ball_angle_bucket in (0, 1, 2, 3, 4, 5, 6, 7), self
        assert isinstance(self.ball_close, bool), self
        assert isinstance(self.closest_to_ball_on_my_team, bool), self
        assert self.opponent_angle_bucket in (0, 1, 2, 3, 4, 5, 6, 7), self
        assert isinstance(self.opponent_close, bool), self
        assert isinstance(self.sticky_actions, tuple), self
        assert len(self.sticky_actions) == 10, self
        # Example of more than 2: sticky_actions=(1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        assert sum(self.sticky_actions[:8]) in [0, 1], self
        assert isinstance(self.run_of_play, bool), self
        assert isinstance(self.am_offside, bool), self
        assert isinstance(self.teammate_offside, bool), self
        assert isinstance(self.role, int), self
        assert 0 <= self.role <= 9, self
        # assert self.role in [e_PlayerRole_CB, e_PlayerRole_CF], self
        return self

# def update_states():
#     agent = Player(
#         player_config = {
#             'index': 0,
#             'player_keyboard': 0,
#             # 'player_ppo2_cnn': 0,
#             'left_players': 1,
#             'right_players': 0,
#         },
#         env_config = {
#             'action_set': 'default',
#             'pitch_scale': 1.0,
#             'warmstart': False,
#             'random_frac': 0.0,
#             'verbose': False,
#             'video': None,
#         })
#     agent.load(checkpoint='agents/agent_3v3_0.5_33.pkl')
#     Q = {}
#
#     n = 0
#     n_values = 0
#     sum_values = 0
#     for state, action_values_dict in agent.Q.items():
#         value = max(action_values_dict.values())
#         n_values += 1
#         sum_values += value
#         # state1 = state._replace(reward=)
#         # state2 = state._replace(role=e_PlayerRole_CF)
#         # Q[state1] = {}
#         # Q[state2] = {}
#         # for action, value in action_values_dict.items():
#         #     if value != 0.0:
#         #         # print('LOADED:', state, action, value)
#         #         Q[state1][action] = value
#         #         Q[state2][action] = value
#         #         n += 2
#     print(sum_values / n_values)
#     # agent.Q = Q
#     # agent.save(checkpoint='agents/agent_3v3_0.5_14.pkl')
