from absl import app
from absl import flags
from absl import logging

from gfootball.env import config
from gfootball.env import football_env

FLAGS = flags.FLAGS

# flags.DEFINE_string('players', 'keyboard:left_players=1',
flags.DEFINE_string('players', 'bot_1v1:left_players=1',
    'Semicolon separated list of players, single keyboard '
    'player on the left by default')
flags.DEFINE_string('level', '1_vs_1_easy', 'Level to play')
flags.DEFINE_enum('action_set', 'default', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
    'If true, environment will slow down so humans can play.')
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')
flags.DEFINE_float('pitch_scale', 0.5, 'Pitch scale. Can be 1.0 or 0.5 for now.')

def main(_):
    players = FLAGS.players.split(';') if FLAGS.players else ''
    assert not (any(['agent' in player for player in players])
                ), ('Player type \'agent\' can not be used with play_game.')
    cfg = config.Config({
        'action_set': FLAGS.action_set,
        'dump_full_episodes': True,
        'players': players,
        'real_time': FLAGS.real_time,
        'pitch_scale': FLAGS.pitch_scale,
    })
    if FLAGS.level:
        cfg['level'] = FLAGS.level
    env = football_env.FootballEnv(cfg)
    if FLAGS.render:
        env.render()
    env.reset()
    try:
        while True:
            obs, reward, done, info = env.step([])
            print(reward, done, info)
            if done:
                env.reset()
    except KeyboardInterrupt:
        logging.warning('Game stopped, writing dump...')
        env.write_dump('shutdown')
        exit(1)

if __name__ == '__main__':
    app.run(main)
