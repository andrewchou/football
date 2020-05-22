import argparse
import logging

from gfootball.common.history import History, HistoryItem
from gfootball.env import config
from gfootball.env import football_env

def bool_arg(x):
    try:
        return bool(int(x))
    except ValueError:
        pass
    x = x.lower()
    if x.startswith('f'):
        return False
    if x.startswith('t'):
        return True
    raise ValueError('Invalid: %s' % x)

def parse_args():
    parser = argparse.ArgumentParser(description='Neural net training')
    # 'keyboard:left_players=1'
    parser.add_argument(
        '--players', type=str, default='bot_1v1:left_players=1',
        help='Semicolon separated list of players, single keyboard player on the left by default')
    parser.add_argument('--level', type=str, default='1_vs_1_easy', help='Level to play')
    parser.add_argument('--action_set', type=str, default='default', help='default or full')
    parser.add_argument('--real_time', type=bool_arg, default=True,
        help='If true, environment will slow down so humans can play.')
    parser.add_argument('--render', type=bool_arg, default=True, help='Whether to do game rendering.')
    parser.add_argument('--warmstart', type=bool_arg, default=False,
        help='Whether to warmstart using the handmade agent.')
    parser.add_argument('--verbose', type=bool_arg, default=True)
    parser.add_argument('--pitch_scale', type=float, default=0.5, help='Pitch scale. Can be 1.0 or 0.5 for now.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Pickle file of Q')
    parser.add_argument('--random_frac', type=float, default=0.1, help='')
    parser.add_argument('--video', type=str, default='', help='')
    parser.add_argument('--num_games', type=int, default=1000000000, help='')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    players = args.players.split(';') if args.players else ''
    # assert not (any(['agent' in player for player in players])
    #             ), ('Player type \'agent\' can not be used with play_game.')
    cfg = config.Config({
        'action_set': args.action_set,
        'dump_full_episodes': True,
        'players': players,
        'real_time': args.real_time and args.render,
        'pitch_scale': args.pitch_scale,
        'warmstart': args.warmstart,
        'random_frac': args.random_frac,
        'verbose': args.verbose,
        'video': args.video,
    })
    if args.level:
        cfg['level'] = args.level
    env = football_env.FootballEnv(cfg)
    if args.checkpoint:
        env._agent.load(checkpoint=args.checkpoint)
    if args.render:
        env.render()
    obs_history = [
        env.reset(),  # Need this to know the initial state
    ]
    self_play_history = History(max_size=int(1e7))
    running_score_update = 0.999
    running_score = [0, 0, 0]
    record = [0, 0, 0]
    try:
        game_num = 0
        # cnts_by_mode = defaultdict(int)
        while True:
            obs, reward, done, info = env.step()
            # _, old_relative_obs = env.get_players_and_relative_obs_pairs(obs=obs_history[-1])
            # _, new_relative_obs = env.get_players_and_relative_obs_pairs(obs=obs)
            if env._agent.num_controlled_right_players() > 0:
                reward *= -1
            item = HistoryItem(
                old_state=obs_history[-1],
                action=info['agent_action'],
                new_state=obs,
                reward=reward.item(),
            )
            self_play_history.add(item=item)
            # cnts_by_mode[(obs[0]['game_mode'], obs[0]['ball_owned_team'])] += 1
            obs_history.append(obs)
            if args.verbose:
                print(reward, done, info)
            if done:
                # defaultdict(<class 'int'>, {(0, -1): 36256, (0, 0): 12701, (0, 1): 55352, (2, -1): 1871, (3, -1): 2146, (5, 1): 140, (5, 0): 19269, (4, -1): 1119, (5, -1): 1, (6, -1): 145})
                # print(cnts_by_mode)
                game_num += 1
                score = obs[0]['score']
                running_score[0] = running_score_update * running_score[0] + (1.0 - running_score_update) * score[0]
                running_score[1] = running_score_update * running_score[1] + (1.0 - running_score_update) * score[1]
                running_score[2] = running_score[0] - running_score[1]
                if score[0] > score[1]:
                    record[0] += 1
                elif score[0] < score[1]:
                    record[2] += 1
                else:
                    record[1] += 1
                # mean_reward = self_play_history.mean_reward()
                print(
                    'Final Score:', score,
                    'Running score: [%.3f, %.3f, %.3f]' % tuple(
                        [x / (1 - running_score_update ** game_num) for x in running_score]),
                    'Record:', record,
                    # 'Mean Reward in history:', mean_reward,
                )
                for item in self_play_history.sample(n=int(1e3)):
                    env._agent.give_reward(item=item) # ._replace(reward=item.reward - mean_reward))
                env.reset()
                if (not args.real_time) and (game_num % 25 == 0):
                    env._agent.save(checkpoint='agent.pkl')
                if game_num == args.num_games:
                    break
    except KeyboardInterrupt:
        logging.warning('Game stopped, writing dump...')
        if (not args.real_time):
            env._agent.save(checkpoint='agent.pkl')
        env.write_dump('shutdown')
        return env._agent
        exit(1)

if __name__ == '__main__':
    # app.run(main)
    main()
    # update_states()
