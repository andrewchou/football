'''Sample bot player.'''

import numpy as np
import pygame

from gfootball.common.colors import RED
from gfootball.common.history import HistoryItem
from gfootball.common.state.angle import relative_angle_bucket
from gfootball.common.writer import Writer, write_text_on_frame
from gfootball.env import football_action_set
from gfootball.env import player_base
from gfootball.policies.base_policy import PolicyConfig, PolicyType
from gfootball.policies.double_expected_sarsa import DoubleExpectedSarsa
from gfootball.policies.nstep_sarsa import NStepSarsa
from gfootball.policies.q_learning import QLearning
from gfootball.scenarios import e_PlayerRole_GK

class BaseRLPlayer(player_base.PlayerBase):
    def __init__(self, player_config, env_config):
        super().__init__(player_config=player_config)
        assert env_config['action_set'] == 'default'
        self.pitch_scale = env_config['pitch_scale']
        assert self.pitch_scale in (1.0, 0.5), self.pitch_scale
        self._observation = None

        self._warmstart = player_config['warmstart']
        assert isinstance(self._warmstart, bool)
        self.verbose = player_config['verbose']
        if player_config['video']:
            self.writer = Writer(filename=player_config['video'])
        else:
            self.writer = None

        self._prev_state = None
        self.policy = self.get_policy(player_config=player_config)

        pygame.init()
        self._init_done = False
        self.reset()

    def get_policy(self, player_config):
        policy_config = player_config['policy_config']
        assert isinstance(policy_config, PolicyConfig), player_config
        if policy_config.policy_type == PolicyType.Q_LEARNING:
            return QLearning(policy_config=policy_config)
        elif policy_config.policy_type == PolicyType.N_STEP_SARSA:
            return NStepSarsa(policy_config=policy_config)
        else:
            assert 0, policy_config
            # return DoubleExpectedSarsa(
            #     random_frac=player_config['random_frac'], checkpoint=player_config['checkpoint'],
            #     verbose=player_config['verbose'],
            # )

    def load(self, checkpoint):
        self.policy.load(checkpoint=checkpoint)

    def save(self, checkpoint):
        self.policy.save(checkpoint=checkpoint)

    def _object_distance(self, object1, object2):
        '''Computes distance between two objects.'''
        return np.linalg.norm(np.array(object1) - np.array(object2))

    def _direction_action(self, delta):
        '''For required movement direction vector returns appropriate action.'''
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
        all_directions_vec = [
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0), (1, -1)]
        all_directions_vec = [
            np.array(v) / np.linalg.norm(np.array(v)) for v in all_directions_vec
        ]
        best_direction = np.argmax([np.dot(delta, v) for v in all_directions_vec])
        action = all_directions[best_direction]
        # TODO
        # if (self._last_action == action): # or (self._last_action == football_action_set.action_sprint):
        #     return football_action_set.action_idle
        return action

    def _closest_opponent_to_object(self, obj):
        '''For a given object returns the closest opponent.

        Args:
          obj: Source object.

        Returns:
          Closest opponent.'''
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            d = self._object_distance(obj, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p
        assert closest is not None
        return closest

    def _closest_team_member_index_to_object(self, o):
        '''For a given object returns the closest team member index.

        Args:
          o: Source object.

        Returns:
          Closest opponent.'''
        min_d = None
        closest_i = None
        for i, p in enumerate(self._observation['left_team']):
            d = self._object_distance(o, p)
            if ((min_d is None) or (d < min_d)) and (self._get_role(index=i) != e_PlayerRole_GK):
                min_d = d
                closest_i = i
        assert closest_i is not None
        return closest_i

    def _closest_team_member_index_to_opponent_with_ball(self):
        return self._closest_team_member_index_to_object(o=self._get_ball_owner_location())

    def _am_closest_team_member_to_opponent_with_ball(self):
        return self._closest_team_member_index_to_object(self._get_ball_owner_location()) == self._get_own_index()

    def _closest_front_opponent(self, o, target):
        '''For an object and its movement direction returns the closest opponent.

        Args:
          o: Source object.
          target: Movement direction.

        Returns:
          Closest front opponent.'''
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

    def _score_pass_target(self, own_position, player_position, debug):
        '''Computes score of the pass between players.

        Args:
          own_position: Player doing the pass.
          player_position: Player receiving the pass.

        Returns:
          Score of the pass.
        '''
        # print('own_position', own_position)
        trajectory = player_position - own_position
        dist_closest_traj = None
        for i in range(10):
            # print('i', i)
            position = own_position + (i + 1) / 10.0 * trajectory
            # print('position', position)
            opp_traj = self._closest_opponent_to_object(position)
            # print('opp_traj', opp_traj)
            dist_traj = self._object_distance(object1=position, object2=opp_traj)
            # print('dist_traj', dist_traj)
            if dist_closest_traj is None or dist_traj < dist_closest_traj:
                dist_closest_traj = dist_traj
        return dist_closest_traj

    def _best_pass_player_index(self, debug):
        '''Computes best pass a given player can do.

        Args:
          own_position: Player doing the pass.

        Returns:
          Best target player receiving the pass.
        '''
        own_position = self._get_own_position()
        best_score = None
        best_target = None
        best_index = None
        for i, player_position in enumerate(self._observation['left_team']):
            if self._object_distance(object1=player_position, object2=own_position) > 0.3:
                continue
            score = self._score_pass_target(
                own_position=own_position, player_position=player_position, debug=debug)
            debug.append(('pass_target', i, 'score', score))
            if best_score is None or score > best_score:
                best_score = score
                best_target = player_position
                best_index = i
        return best_index, best_target

    def _get_own_keeper_index(self):
        for index in range(len(self._observation['left_team'])):
            role = self._get_role(index=index)
            if role == e_PlayerRole_GK:
                return index
        assert 0, self._observation

    def _avoid_opponent(self, active, opponent, target):
        '''Computes movement action to avoid a given opponent.

        Args:
          active: Active player.
          opponent: Opponent to be avoided.
          target: Original movement direction of the active player.

        Returns:
          Action to perform to avoid the opponent.
        '''
        # Choose a perpendicular direction to the opponent, towards the target.
        delta = opponent - active
        delta_t = target - active
        new_delta = [delta[1], -delta[0]]
        if delta_t[0] * new_delta[0] < 0:
            new_delta = [-new_delta[0], -new_delta[1]]

        return self._direction_action(new_delta)

    def _get_ball_location(self):
        return self._observation['ball'][:2]

    def _get_ball_owner_location(self):
        assert self._observation['ball_owned_team'] == 1, self._observation
        other_team_positions = self._observation['right_team']
        other_player_with_ball_position = other_team_positions[self._observation['ball_owned_player']]
        return other_player_with_ball_position

    def _get_ball_owner_location_target(self):
        return self._get_ball_owner_location() - self._get_own_position()

    def _get_own_position(self):
        return self._observation['left_team'][self._get_own_index()]

    def _they_have_the_ball(self):
        return self._observation['ball_owned_team'] == 1

    def _their_gk_has_the_ball(self):
        if not self._they_have_the_ball():
            return False
        other_role = self._observation['right_team_roles'][self._observation['ball_owned_player']]
        return other_role == e_PlayerRole_GK

    def _we_have_the_ball(self):
        return self._observation['ball_owned_team'] == 0

    def _get_sticky_actions(self):
        return self._observation['sticky_actions']

    def _get_own_index(self):
        return self._observation['active']

    def _get_hardcoded_action(self, debug):
        '''Returns action to perform for the current observations.'''
        assert 0, 'Need to implement _get_hardcoded_action()'

    def _opponent_angle_bucket_relative_to_me(self):
        position = self._get_own_position()
        opponent = self._closest_opponent_to_object(obj=position)
        return self._angle_bucket_relative_to_me(other_position=opponent)

    def _ball_angle_bucket_relative_to_me(self):
        return self._angle_bucket_relative_to_me(other_position=self._get_ball_location())

    def _angle_bucket_relative_to_me(self, other_position):
        position = self._get_own_position()
        delta = other_position - position
        return relative_angle_bucket(delta_position=delta)

    def _opponent_distance_from_me(self):
        position = self._get_own_position()
        opponent = self._closest_opponent_to_object(obj=position)
        return self._object_distance(object1=position, object2=opponent)

    def _ball_distance_from_me(self):
        position = self._get_own_position()
        return self._object_distance(object1=position, object2=self._get_ball_location())

    def _am_offside(self):
        own_pos = self._get_own_position()
        num_opponents_closer_to_their_goal = 0
        for opponent_pos in self._observation['right_team']:
            if own_pos[0] < opponent_pos[0]:
                num_opponents_closer_to_their_goal += 1
        return bool(
            (num_opponents_closer_to_their_goal < 2) and (own_pos[0] > 0)
        )

    def _get_role(self, index=None):
        if index is None:
            index = self._get_own_index()
        return int(self._observation['left_team_roles'][index])

    def get_state(self, observations):
        assert len(observations) == 1, len(observations)
        assert 0, 'Need to implement get_state()'

    def give_reward(self, item):
        assert isinstance(item, HistoryItem), item
        self.policy.give_reward(item=item._replace(
            old_state=self.get_state(observations=item.old_state),
            new_state=self.get_state(observations=item.new_state),
        ))

    def process_epoch(self, items):
        new_items = []
        for item in items:
            assert isinstance(item, HistoryItem), item
            new_items.append(item._replace(
                old_state=self.get_state(observations=item.old_state),
                new_state=self.get_state(observations=item.new_state),
            ))
        self.policy.process_epoch(items=new_items)

    def take_action(self, observations):
        if not self._init_done:
            self._init_done = True
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
        assert len(observations) == 1, 'Bot does not support multiple player control'
        if self.verbose:
            print()

        # print('OBS')
        # for k, v in sorted(observations[0].items()):
        #     if k != 'frame':
        #         print(k, v)
        self._observation = observations[0]
        state = self.get_state(observations=observations)
        if self.writer:
            frame = self._observation['frame'][:, :, [2, 1, 0]].copy()
            for i, (k, v) in enumerate(sorted(state._asdict().items())):
                write_text_on_frame(
                    frame=frame, text='%s: %s' % (k, v),
                    color=RED, bottom_left_corner_of_text=(30, 20 * (i + 2)),
                    thickness=1, font_scale=0.5)
            for i, (k, v) in enumerate(sorted(self._observation.items())):
                if k == 'frame':
                    continue
                write_text_on_frame(
                    frame=frame, text='%s: %s' % (k, v),
                    color=RED, bottom_left_corner_of_text=(360, 20 * (i + 2)),
                    thickness=1, font_scale=0.5)
        if self.verbose:
            print(state)
        debug = []
        if self._warmstart:
            action = self._get_hardcoded_action(debug=debug)
            debug.append(('Warmstart action:', action))
        else:
            action = self.policy.get_action(state=state, debug=debug)
        if self.writer:
            for i, x in enumerate(debug):
                write_text_on_frame(
                    frame=frame, text=str(x),
                    color=RED, bottom_left_corner_of_text=(20, 720 - 20 * (len(debug) - i)),
                    thickness=1, font_scale=0.5)
            self.writer.write(frame=frame)
        self._action_history.append(action)
        return [action]

    def reset(self):
        self._action_history = []
