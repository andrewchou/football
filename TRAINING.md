# Warmstart an agent using hardcoded actions
python gfootball/train_basic.py --players agent_rl_1v1:left_players=1 --random_frac 0.1 --render 0 --real_time 0 --verbose 0 --pitch_scale 0.5 --warmstart 1
# It will output agent.pkl. Pass that in as the checkpoint and turn off --warmstart.
python gfootball/train_basic.py --players agent_rl_1v1:left_players=1 --random_frac 0.1 --render 0 --real_time 0 --verbose 0 --pitch_scale 0.5 --warmstart 0 --checkpoint agent20.pkl
# Turn down --random_frac to make it play better.

# 3v3
python gfootball/train_basic.py --players agent_rl_3v3:left_players=1 --level 3_vs_3 --random_frac 0.1 --render 0 --real_time 0 --verbose 1 --pitch_scale 0.5 --warmstart 1
