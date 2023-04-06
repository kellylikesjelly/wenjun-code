#!/bin/bash
#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

# How many nodes you require? You should only require 1.
#SBATCH --nodes=1

# When should you receive an email? sbatch --help for more options 
#SBATCH --mail-type=BEGIN,END,FAIL

# Number of CPU you'll like
#SBATCH --cpus-per-task=16

# Memory
#SBATCH --mem 8GB

# How long do you require the resources? Note : If the job exceeds this run time, the manager will kill the job.
#SBATCH --time=5-00:00:00

# Do you require GPUS? If not, you should remove the following line (max GPU 1)
#SBATCH --gres=gpu:1

# Where should the log files go?
# You must provide an absolute path eg /common/home/module/username/
# If no paths are provided, the output file will be placed in your current working directory
#SBATCH --output=%u.%j.out # STDOUT
#SBATCH --prefer=v100

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################
# The account you've been assigned (normally student)
#SBATCH --account=pradeepresearch

# The partition you've been assigned
#SBATCH --partition=researchlong

#What is the QOS assigned to you? Check with myinfo command
#SBATCH --qos=wjli-20230101

# Who should receive the email notifications
#SBATCH --mail-user=wjli.2020@phdcs.smu.edu.sg

# Give the job a name
#SBATCH --job-name=bw_paired_seed93

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################
# Purge the enviromnent, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
ml load cuDNN/8.1.1.33-CUDA-11.2.2
module load Anaconda3/2022.05

#We need the eval to have conda recognise the path.
eval "$(conda shell.bash hook)"
conda activate dcd2
echo "activated"

srun --chdir=/common/home/users/w/wjli.2020/env_generation/bw_paired_seed93/ --cpus-per-task=16 --gres=gpu:1 python3 -u -m train --xpid=ued-BipedalWalker-Adversarial-v0-paired-lr0.0003-epoch5-mb32-v0.5-gc0.5-henv0.01-ha0.001-tl_0 --env_name=BipedalWalker-Adversarial-v0 --use_gae=True --gamma=0.99 --gae_lambda=0.9 --seed=9 --num_control_points=12 --recurrent_arch=lstm --recurrent_agent=False --recurrent_adversary_env=False --recurrent_hidden_size=1 --use_global_critic=False --lr=0.0003 --num_steps=2048 --num_processes=16 --num_env_steps=1000000000 --ppo_epoch=5 --num_mini_batch=32 --entropy_coef=0.001 --value_loss_coef=0.5 --clip_param=0.2 --clip_value_loss=False --adv_entropy_coef=0.01 --max_grad_norm=0.5 --algo=ppo --ued_algo=paired --use_plr=False --level_replay_prob=0.0 --level_replay_rho=0.5 --level_replay_seed_buffer_size=1000 --level_replay_score_transform=rank --level_replay_temperature=0.1 --staleness_coef=0.5 --no_exploratory_grad_updates=False --use_editor=False --level_editor_prob=0 --level_editor_method=random --num_edits=0 --base_levels=batch --log_interval=10 --screenshot_interval=200 --log_grad_norm=True --normalize_returns=True --checkpoint_basis=student_grad_updates --archive_interval=5000 --reward_shaping=True --use_categorical_adv=True --use_skip=False --choose_start_pos=False --sparse_rewards=False --handle_timelimits=True --adv_max_grad_norm=0.5 --adv_ppo_epoch=8 --adv_num_mini_batch=4 --adv_normalize_returns=False --adv_use_popart=False --level_replay_strategy=positive_value_loss --test_env_names=BipedalWalker-v3,BipedalWalkerHardcore-v3,BipedalWalker-Med-Stairs-v0,BipedalWalker-Med-PitGap-v0,BipedalWalker-Med-StumpHeight-v0,BipedalWalker-Med-Roughness-v0 --log_dir=./logs/data_seed93/ --test_interval=100 --test_num_episodes=10 --test_num_processes=2 --log_plr_buffer_stats=True --log_replay_complexity=True --checkpoint=True --log_action_complexity=False




