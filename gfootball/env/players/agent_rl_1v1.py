"""Sample bot player."""
import pickle
import random
from collections import defaultdict, namedtuple

import numpy as np
import pygame

from gfootball.env import football_action_set
from gfootball.env import player_base
from gfootball.env.football_action_set import DEFAULT_ACTION_SET

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

        self._warmstart = env_config['warmstart']
        assert isinstance(self._warmstart, bool)
        self.random_frac = env_config['random_frac']
        self.verbose = env_config['verbose']

        self._prev_state = None
        self.Q = defaultdict(lambda: defaultdict(float))
        # self._state_values = defaultdict(float)
        # self._transitions_counts = defaultdict(lambda : defaultdict(int))

        pygame.init()
        self._init_done = False

        # self._action = None

    def load(self, checkpoint):
        with open(checkpoint, 'rb') as inf:
            Q = pickle.load(inf)
        n = 0
        for state, action_values_dict in Q.items():
            for action, value in action_values_dict.items():
                if value != 0.0:
                    # print('LOADED:', state, action, value)
                    self.Q[state][action] = value
                    n += 1
        print('Loaded %d Q values.' % n)

    def save(self, checkpoint):
        Q = {k: {k2: v2 for k2, v2 in v.items() if v2 != 0.0} for k, v in self.Q.items()}
        n = sum([len(v) for v in Q.values()])
        with open(checkpoint, 'wb') as f:
            pickle.dump(Q, f)
        print('Saved %d Q values.' % n)

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
            if self.verbose: print('Not in the run of play. Mode:', self._observation['game_mode'])
            # print(self._observation)
            sticky_actions = self._get_sticky_actions()
            if not sticky_actions[4]:
                return football_action_set.action_right
            return football_action_set.action_long_pass

        #
        if self._they_have_the_ball():
            if self.verbose:
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
        if self.verbose:
            print('distance_from_good_spot_to_shoot_from', distance_from_good_spot_to_shoot_from,
            self._get_ball_location_target(), GOOD_SPOT_TO_SHOOT_FROM)
        if distance_from_good_spot_to_shoot_from < self._shoot_distance:
            if self.verbose:
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

    def _opponent_angle_bucket_relative_to_me(self):
        position = self._get_own_position()
        opponent = self._closest_opponent_to_object(o=position)
        return self._angle_bucket_relative_to_me(other_position=opponent)

    def _ball_angle_bucket_relative_to_me(self):
        return self._angle_bucket_relative_to_me(other_position=self._get_ball_location_target())

    def _angle_bucket_relative_to_me(self, other_position):
        position = self._get_own_position()
        delta = other_position - position
        # Angle in radians, in the range [-pi, pi].
        radians = np.arctan2(delta[1], delta[0])
        degrees = radians * 180 / np.pi
        assert -180 <= degrees <= 180, degrees
        degrees_plus_210 = degrees + 210  # [30, 390]
        assert 30 <= degrees_plus_210 <= 390, degrees_plus_210
        wrapped_degrees_plus_210 = degrees_plus_210 % 360
        assert 0 <= wrapped_degrees_plus_210 <= 360, wrapped_degrees_plus_210
        bucket = wrapped_degrees_plus_210 // 60  #
        if bucket == 6:
            assert wrapped_degrees_plus_210 == 360, wrapped_degrees_plus_210
            bucket = 5
        assert 0 <= bucket <= 5, bucket
        bucket = (bucket + 3) % 6
        # print('BUCKET')
        # print(position)
        # print(opponent)
        # print(delta)
        # print(degrees)
        # print(bucket)
        assert int(bucket) == bucket, bucket
        return int(bucket)

    def _opponent_distance_from_me(self):
        position = self._get_own_position()
        opponent = self._closest_opponent_to_object(o=position)
        return self._object_distance(object1=position, object2=opponent)

    def _ball_distance_from_me(self):
        position = self._get_own_position()
        return self._object_distance(object1=position, object2=self._get_ball_location_target())

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

    # def set_action(self, action):
    #     self._action = action

    def give_reward(self, old_relative_obs, action, new_relative_obs, reward):
        # assert isinstance(old_state,)
        assert self._last_action is not None
        old_state = self.get_state(observations=old_relative_obs)
        new_state = self.get_state(observations=new_relative_obs)
        possible_actions_dict = self.Q[new_state]
        best_action_value = max(possible_actions_dict.values()) if possible_actions_dict else 0
        alpha = 0.01
        discount = 0.999
        self.Q[old_state][action] = (
            (1.0 - alpha) * self.Q[old_state][action] +
            alpha * (reward.item() + discount * best_action_value)
        )
        assert isinstance(self.Q[old_state][action], float), self.Q[old_state][action]
        # if reward.item() != 0:
        #     assert 0, 'TODO'

    def take_action(self, observations):
        if not self._init_done:
            self._init_done = True
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
        assert len(observations) == 1, 'Bot does not support multiple player control'
        if self.verbose: print()

        # print('OBS')
        # for k, v in sorted(observations[0].items()):
        #     if k != 'frame':
        #         print(k, v)
        self._observation = observations[0]
        if self._warmstart:
            action = self._get_action()
            if self.verbose:
                print('Warmstart action:', action)
        else:
            state = self.get_state(observations=observations)
            if self.verbose:
                print(state)
            possible_actions_dict = self.Q[state]
            if (not possible_actions_dict) or (random.random() < self.random_frac):
                action = DEFAULT_ACTION_SET[random.randint(0, len(DEFAULT_ACTION_SET) - 1)]
                if self.verbose:
                    print('Random action:', action)
            else:
                assert possible_actions_dict
                for a in DEFAULT_ACTION_SET:
                    if a not in possible_actions_dict:
                        possible_actions_dict[a] = 0.0
                best_value = max(possible_actions_dict.values())
                top_actions = [a for a in DEFAULT_ACTION_SET if possible_actions_dict[a] == best_value]
                action = top_actions[random.randint(0, len(top_actions) - 1)]
                if self.verbose:
                    print(' Value action:', action, best_value, min(possible_actions_dict.values()))
                assert isinstance(best_value, float)
        self._last_action = action
        # self._prev_state = cur_state
        # assert self._last_action in ACTION_SET_DICT['default'], self._last_action
        # print(self._last_action)
        return [self._last_action]

class BasicState(namedtuple('BasicState', [
    'ball_owned_team',  # 3
    'field_position',  # 6 spots (front/back, right/left/center)
    'ball_angle_bucket',  # 6 (60 degree buckets)
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
        assert self.ball_angle_bucket in (0, 1, 2, 3, 4, 5), self
        assert isinstance(self.ball_close, bool), self
        assert self.opponent_angle_bucket in (0, 1, 2, 3, 4, 5), self
        assert isinstance(self.opponent_close, bool), self
        assert isinstance(self.sticky_actions, tuple), self
        assert len(self.sticky_actions) == 10, self
        # Example of more than 2: sticky_actions=(1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        # assert sum(self.sticky_actions) in [0, 1, 2], self
        assert isinstance(self.run_of_play, bool), self
        return self
