import argparse
import logging

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
    self_play_history = []
    running_score_update = 0.99
    running_score = [0, 0, 0]
    record = [0, 0, 0]
    try:
        game_num = 0
        while True:
            obs, reward, done, info = env.step()
            # _, old_relative_obs = env.get_players_and_relative_obs_pairs(obs=obs_history[-1])
            # _, new_relative_obs = env.get_players_and_relative_obs_pairs(obs=obs)
            if env._agent.num_controlled_right_players() > 0:
                reward *= -1
            env._agent.give_reward(
                # old_relative_obs=old_relative_obs,
                old_relative_obs=obs_history[-1],
                action=info['agent_action'],
                # new_relative_obs=new_relative_obs,
                new_relative_obs=obs,
                reward=reward)
            self_play_history.append((
                obs_history[-1],
                info['agent_action'],
                obs,
                reward,
            ))
            obs_history.append(obs)
            if args.verbose:
                print(reward, done, info)
            if done:
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
                print('Final Score:', score, 'Running score: [%.3f, %.3f, %.3f]' % tuple(running_score), 'Record:', record)
                for (old_relative_obs, action, new_relative_obs, reward) in reversed(self_play_history):
                    env._agent.give_reward(
                        old_relative_obs=old_relative_obs,
                        action=action,
                        new_relative_obs=new_relative_obs,
                        reward=reward)
                env.reset()
                self_play_history = []
                if (not args.real_time) and (game_num % 10 == 0):
                    env._agent.save(checkpoint='agent.pkl')
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