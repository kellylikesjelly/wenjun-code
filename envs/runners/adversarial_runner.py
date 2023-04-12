# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from email.mime import image
import os
from collections import deque, defaultdict
from tkinter.tix import MAX

import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd

from torch.utils.tensorboard import SummaryWriter

from level_replay import LevelSampler, LevelStore
from util import \
    array_to_csv, \
    is_discrete_actions, \
    get_obs_at_index, \
    set_obs_at_index

from teachDeepRL.teachers.teacher_controller import TeacherController
from util import save_images

import matplotlib as mpl
import matplotlib.pyplot as plt
from util import global_stats
import sys
from geomloss import SamplesLoss
import geomloss
import torch.nn as nn
import time


'''
def cosine_cost(a, b, p=2):
    """ a, b in shape: (B, N, D) or (N, D)
    """ 
    a = a.squeeze()
    b = b.squeeze()
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if a.dim() == 3:
        x_norm = a / a.norm(dim=2)[:, :, None]
        y_norm = b / b.norm(dim=2)[:, :, None]
        M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
    elif a.dim() == 2:
        x_norm = a / a.norm(dim=1)[:, None]
        y_norm = b / b.norm(dim=1)[:, None]
        # print('x_norm and y_norm are computed..')
        # print('x_norm={} and y_norm={}'.format(x_norm, y_norm))
        M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        # print('x_norm.shape={}, y_norm.shape={}, M.shape={}'.format(x_norm.shape, y_norm.shape, M.shape))
    M = pow(M, p)
    print('cost_matrix={}'.format(M))
    return M
'''


class AdversarialRunner(object):
    """
    Performs rollouts of an adversarial environment, given 
    protagonist (agent), antogonist (adversary_agent), and
    environment adversary (advesary_env)
    """
    def __init__(
        self,
        args,
        venv,
        agent,
        ued_venv=None,
        adversary_agent=None,
        adversary_env=None,
        flexible_protagonist=False,
        train=False,
        plr_args=None,
        device='cpu'):
        """
        venv: Vectorized, adversarial gym env with agent-specific wrappers.
        agent: Protagonist trainer.
        ued_venv: Vectorized, adversarial gym env with adversary-env-specific wrappers.
        adversary_agent: Antogonist trainer.
        adversary_env: Environment adversary trainer.

        flexible_protagonist: Which agent plays the role of protagonist in
            calculating the regret depends on which has the lowest score.
        """
        self.args = args

        self.venv = venv
        if ued_venv is None:
            self.ued_venv = venv
        else:
            self.ued_venv = ued_venv # Since adv env can have different env wrappers

        self.is_discrete_actions = is_discrete_actions(self.venv)
        self.is_discrete_adversary_env_actions = is_discrete_actions(self.venv, adversary=True)

        self.agents = {
            'agent': agent,
            'adversary_agent': adversary_agent,
            'adversary_env': adversary_env,
        }

        self.agent_rollout_steps = args.num_steps
        self.adversary_env_rollout_steps = self.venv.adversary_observation_space['time_step'].high[0]

        self.is_dr = args.ued_algo == 'domain_randomization'
        self.is_training_env = args.ued_algo in ['paired', 'flexible_paired', 'minimax']
        self.is_paired = args.ued_algo in ['paired', 'flexible_paired']
        self.requires_batched_vloss = args.use_editor and args.base_levels == 'easy'

        self.is_alp_gmm = args.ued_algo == 'alp_gmm' 

        # Track running mean and std of env returns for return normalization
        if args.adv_normalize_returns:
            self.env_return_rms = RunningMeanStd(shape=())

        self.device = device

        if train:
            self.train()
        else:
            self.eval()

        self.reset()

        # Set up PLR
        self.level_store = None
        self.level_samplers = {}
        self.current_level_seeds = None
        self.weighted_num_edits = 0
        self.latest_env_stats = defaultdict(float)
        if plr_args:
            if self.is_paired:
                if not args.protagonist_plr and not args.antagonist_plr:
                    self.level_samplers.update({
                        'agent': LevelSampler(**plr_args),
                        'adversary_agent': LevelSampler(**plr_args)
                    })
                elif args.protagonist_plr:
                    self.level_samplers['agent'] = LevelSampler(**plr_args)
                elif args.antagonist_plr:
                    self.level_samplers['adversary_agent'] = LevelSampler(**plr_args)
            else:
                self.level_samplers['agent'] = LevelSampler(**plr_args)

            if self.use_byte_encoding:
                example = self.ued_venv.get_encodings()[0]
                data_info = {
                    'numpy': True,
                    'dtype': example.dtype,
                    'shape': example.shape
                }
                self.level_store = LevelStore(data_info=data_info)
            else:
                self.level_store = LevelStore()

            self.current_level_seeds = [-1 for i in range(args.num_processes)]

            self._default_level_sampler = self.all_level_samplers[0]

            self.use_editor = args.use_editor
            self.edit_prob = args.level_editor_prob
            self.base_levels = args.base_levels
        else:
            self.use_editor = False
            self.edit_prob = 0
            self.base_levels = None

        # Set up ALP-GMM
        if self.is_alp_gmm:
            self._init_alp_gmm()

        # wenjun: 2022.nov.11
        '''
        self.wenjun_screenshot_dir = "./screenshots/"
        if not os.path.exists(self.wenjun_screenshot_dir):
            os.makedirs(self.wenjun_screenshot_dir)
        self.current_epi = 0
        self.replaybuffer_passable_ratio = 0.0        
        self.replaybuffer_num_blocks = 0.0
        self.replaybuffer_shortest_path_length = 0.0
        '''

        # ====================== Euclidean(Params) ====================== # 
        '''
        # wenjun: 2022.dec, deversity
        if self.args.env_name.startswith('BipedalWalker'):
            self.env_params_in_buffer = torch.zeros((self.args.level_replay_seed_buffer_size, 8), device='cuda')
            self.env_params = torch.zeros((self.args.level_replay_seed_buffer_size, 8), device='cuda')              # for any envs, either new or in_buffer
        elif self.args.env_name.startswith('MultiGrid'):
            self.env_params_in_buffer = torch.ones((62, self.args.level_replay_seed_buffer_size), device='cuda') * 9
            self.env_params = torch.zeros((62, self.args.level_replay_seed_buffer_size), device='cuda') * 9             # for any envs, either new or in_buffer

        # self.level_diversity_for_levels_in_buffer = torch.zeros((self.args.num_processes, 1), device='cuda')
        # self.level_diversity_for_levels_in_new = torch.zeros((self.args.num_processes, 1), device='cuda')
        self.level_diversity_for_any_levels = torch.ones((self.args.num_processes, 1), device='cuda')              # assign high value when init
        self.level_diversity_sum = 0        # sum of diversity in the buffer

        self.PARAM_RANGES_FULL = torch.Tensor([10, 10, 10, 5, 5, 5, 5, 8]).cuda()
        # PARAM_RANGES_FULL = {
        #     1: [0,10], # ground roughness
        #     2: [0,10], # pit gap 1
        #     3: [0,10], # pit gap 2
        #     4: [0,5], # stump height 1
        #     5: [0,5], # stump height 2
        #     6: [0,5], # stair height 1
        #     7: [0,5], # stair height 2
        #     8: [1,9], # stair steps
        # }

        # params needed for parallel
        self.NUM_SPLIT = int(self.args.level_replay_seed_buffer_size / self.args.num_processes)

        # params needed for minigrid diversity
        self.width = 15
        self.adversary_action_dim = 169
        self.n_clutter = 60
        '''
        # ====================== Euclidean(Params) ====================== # 





        # ====================== Euclidean(Trajs) ====================== # 
        # variables for store trajs on levels in the buffer
        if args.env_name.startswith('BipedalWalker'):
            obs_dim = 24
            self.OBS_NORMALIZE = torch.Tensor([3.14, 5, 5, 5, 3.14, 5, 3.14, 5, 5, 3.14, 5, 3.14, 5, 5,
                                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]) 

        # self.Buffer_Trajs = torch.zeros((args.num_steps + 1, args.num_processes, obs_dim), device='cuda')
        self.Buffer_Trajs = torch.zeros((args.num_steps + 1, args.num_processes, obs_dim), device='cpu')
        # self.New_Trajs = torch.zeros((args.num_steps + 1, args.num_processes, obs_dim), device='cuda')
        self.New_Trajs = torch.zeros((args.num_steps + 1, args.num_processes, obs_dim), device='cpu')
        self.level_divgence = torch.ones(args.num_processes, 1)
        
        self.divergence_sum = 0
        self.ever_replayed = False
        self.current_epi = 0

        # self.div_definition = 'avg'   
        self.div_definition = 'min'
        # self.div_definition = 'maxmin'

        # hyper-params for div
        self.blur = 0.1
        self.p = 2
        

        # flag: collect trajs on replay levels when student policy updated, otherwise, no need
        self.student_policy_updated = False

        # ====================== Euclidean(Trajs) ====================== # 


        ''' variables for debug
        self.current_epi = 0
        self.level_replay
        self.wenjun_screenshot_dir = "./screenshots/"
        if not os.path.exists(self.wenjun_screenshot_dir):
            os.makedirs(self.wenjun_screenshot_dir)
        '''

    @property
    def use_byte_encoding(self):
        env_name = self.args.env_name
        if self.args.use_editor \
           or env_name.startswith('BipedalWalker') \
           or (env_name.startswith('MultiGrid') and self.args.use_reset_random_dr):
            return True
        else:
            return False

    def _init_alp_gmm(self):
        args = self.args
        param_env_bounds = []
        if args.env_name.startswith('MultiGrid'):
            param_env_bounds = {'actions':[0,168,26]}
            reward_bounds = None
        elif args.env_name.startswith('Bipedal'):
            if 'POET' in args.env_name:
                param_env_bounds = {'actions': [0,2,5]}
            else:
                param_env_bounds = {'actions': [0,2,8]}
            reward_bounds = (-200, 350)
        else:
            raise ValueError(f'Environment {args.env_name} not supported for ALP-GMM')

        self.alp_gmm_teacher = TeacherController(
                    teacher='ALP-GMM',
                    nb_test_episodes=0,
                    param_env_bounds=param_env_bounds,
                    reward_bounds=reward_bounds,
                    seed=args.seed,
                    teacher_params={}) # Use defaults

    def reset(self):
        self.num_updates = 0
        self.total_num_edits = 0
        self.total_episodes_collected = 0
        self.total_seeds_collected = 0
        self.student_grad_updates = 0
        self.sampled_level_info = None

        max_return_queue_size = 10
        self.agent_returns = deque(maxlen=max_return_queue_size)
        self.adversary_agent_returns = deque(maxlen=max_return_queue_size)

    def train(self):
        self.is_training = True
        [agent.train() if agent else agent for _,agent in self.agents.items()]

    def eval(self):
        self.is_training = False
        [agent.eval() if agent else agent for _,agent in self.agents.items()]

    def state_dict(self):
        agent_state_dict = {}
        optimizer_state_dict = {}
        for k, agent in self.agents.items():
            if agent:
                agent_state_dict[k] = agent.algo.actor_critic.state_dict()
                optimizer_state_dict[k] = agent.algo.optimizer.state_dict()

        return {
            'agent_state_dict': agent_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'agent_returns': self.agent_returns,
            'adversary_agent_returns': self.adversary_agent_returns,
            'num_updates': self.num_updates,
            'total_episodes_collected': self.total_episodes_collected,
            'total_seeds_collected': self.total_seeds_collected,
            'total_num_edits': self.total_num_edits,
            'student_grad_updates': self.student_grad_updates,
            'latest_env_stats': self.latest_env_stats,
            'level_store': self.level_store,
            'level_samplers': self.level_samplers,
        }

    def load_state_dict(self, state_dict):

        agent_state_dict = state_dict.get('agent_state_dict')

        for k,state in agent_state_dict.items():
            self.agents[k].algo.actor_critic.load_state_dict(state)

        optimizer_state_dict = state_dict.get('optimizer_state_dict')

        for k, state in optimizer_state_dict.items():
            self.agents[k].algo.optimizer.load_state_dict(state)

        self.agent_returns = state_dict.get('agent_returns')
        self.adversary_agent_returns = state_dict.get('adversary_agent_returns')
        self.num_updates = state_dict.get('num_updates')
        self.total_episodes_collected = state_dict.get('total_episodes_collected')
        self.total_seeds_collected = state_dict.get('total_seeds_collected')
        self.total_num_edits = state_dict.get('total_num_edits')
        self.student_grad_updates = state_dict.get('student_grad_updates')
        self.latest_env_stats = state_dict.get('latest_env_stats')

        self.level_store = state_dict.get('level_store')
        self.level_samplers = state_dict.get('level_samplers')

        if self.args.use_plr:
            self._default_level_sampler = self.all_level_samplers[0]

            if self.use_editor:
                self.weighted_num_edits = self._get_weighted_num_edits()

    def _get_batched_value_loss(self, agent, clipped=True, batched=True):
        batched_value_loss = agent.storage.get_batched_value_loss(
            signed=False, 
            positive_only=False, 
            clipped=clipped,
            batched=batched)

        return batched_value_loss

    def _get_rollout_return_stats(self, rollout_returns):
        mean_return = torch.zeros(self.args.num_processes, 1)
        max_return = torch.zeros(self.args.num_processes, 1)
        for b, returns in enumerate(rollout_returns):
            if len(returns) > 0:
                mean_return[b] = float(np.mean(returns))
                max_return[b] = float(np.max(returns))

        stats = {
            'mean_return': mean_return,
            'max_return': max_return,
            'returns': rollout_returns 
        }

        return stats

    def _get_env_stats_multigrid(self, agent_info, adversary_agent_info):
        num_blocks = np.mean(self.venv.get_num_blocks())
        passable_ratio = np.mean(self.venv.get_passable())
        shortest_path_lengths = self.venv.get_shortest_path_length()
        shortest_path_length = np.mean(shortest_path_lengths)

        if 'max_returns' in adversary_agent_info:
            solved_idx = \
                (torch.max(agent_info['max_return'], \
                    adversary_agent_info['max_return']) > 0).numpy().squeeze()
        else:
            solved_idx = (agent_info['max_return'] > 0).numpy().squeeze()

        solved_path_lengths = np.array(shortest_path_lengths)[solved_idx]
        solved_path_length = np.mean(solved_path_lengths) if len(solved_path_lengths) > 0 else 0

        stats = {
            'num_blocks': num_blocks,
            'passable_ratio': passable_ratio,
            'shortest_path_length': shortest_path_length,
            'solved_path_length': solved_path_length
        }

        return stats

    def _get_replaybuffer_stats_multigrid(self, ):
        num_blocks = np.mean(self.venv.get_num_blocks())
        passable_ratio = np.mean(self.venv.get_passable())
        shortest_path_lengths = self.venv.get_shortest_path_length()
        shortest_path_length = np.mean(shortest_path_lengths)
        return num_blocks, passable_ratio, shortest_path_length

    def _get_plr_buffer_stats(self):
        stats = {}
        for k,sampler in self.level_samplers.items():
            stats[k + '_plr_passable_mass'] = sampler.solvable_mass
            stats[k + '_plr_max_score'] = sampler.max_score 
            stats[k + '_plr_weighted_num_edits'] = self.weighted_num_edits

        return stats

    def _get_env_stats_car_racing(self, agent_info, adversary_agent_info):
        infos = self.venv.get_complexity_info()
        num_envs = len(infos)

        sums = defaultdict(float)
        for info in infos:
            for k,v in info.items():
                sums[k] += v

        stats = {}
        for k,v in sums.items():
            stats['track_' + k] = sums[k]/num_envs

        return stats

    def _get_env_stats_bipedalwalker(self, agent_info, adversary_agent_info):
        infos = self.venv.get_complexity_info()
        num_envs = len(infos)

        sums = defaultdict(float)
        for info in infos:
            for k,v in info.items():
                sums[k] += v

        stats = {}
        for k,v in sums.items():
            stats['track_' + k] = sums[k]/num_envs

        return stats

    def _get_env_stats(self, agent_info, adversary_agent_info, log_replay_complexity=False):
        env_name = self.args.env_name
        if env_name.startswith('MultiGrid'):
            stats = self._get_env_stats_multigrid(agent_info, adversary_agent_info)
        elif env_name.startswith('CarRacing'):
            stats = self._get_env_stats_car_racing(agent_info, adversary_agent_info)
        elif env_name.startswith('BipedalWalker'):
            stats = self._get_env_stats_bipedalwalker(agent_info, adversary_agent_info)
        else:
            raise ValueError(f'Unsupported environment, {self.args.env_name}')

        stats_ = {}
        for k,v in stats.items():
            stats_['plr_' + k] = v if log_replay_complexity else None
            stats_[k] = v if not log_replay_complexity else None
            
        return stats_

    def _get_active_levels(self):
        assert self.args.use_plr, 'Only call _get_active_levels when using PLR.'

        env_name = self.args.env_name

        is_multigrid = env_name.startswith('MultiGrid')
        is_car_racing = env_name.startswith('CarRacing')
        is_bipedal_walker = env_name.startswith('BipedalWalker')

        if self.use_byte_encoding:
            # print('params_vec={}, after encoding={}'.format(self.ued_venv.get_encodings(), [x.tobytes() for x in self.ued_venv.get_encodings()]))
            # wenjun: this transform the params_vec to bytes
            return [x.tobytes() for x in self.ued_venv.get_encodings()]
        elif is_multigrid:
            return self.agents['adversary_env'].storage.get_action_traj(as_string=True)
        else:
            return self.ued_venv.get_level()

    def _get_level_sampler(self, name):
        other = 'adversary_agent'
        if name == 'adversary_agent':
            other = 'agent'

        level_sampler = self.level_samplers.get(name) or self.level_samplers.get(other)

        updateable = name in self.level_samplers

        return level_sampler, updateable

    @property
    def all_level_samplers(self):
        if len(self.level_samplers) == 0:
            return []

        return list(filter(lambda x: x is not None, [v for _, v in self.level_samplers.items()]))

    def _should_edit_level(self):
        if self.use_editor:
            return np.random.rand() < self.edit_prob
        else:
            return False

    def _update_plr_with_current_unseen_levels(self, parent_seeds=None):
        args = self.args
        levels = self._get_active_levels()                              

        self.current_level_seeds = self.level_store.insert(levels, parent_seeds=parent_seeds)
        if args.log_plr_buffer_stats or args.reject_unsolvable_seeds:
            passable = self.venv.get_passable()
        else:
            passable = None
        self._update_level_samplers_with_external_unseen_sample(self.current_level_seeds, solvable=passable)

    def _update_level_samplers_with_external_unseen_sample(self, seeds, solvable=None):
        level_samplers = self.all_level_samplers

        if self.args.reject_unsolvable_seeds:
            solvable = np.array(solvable, dtype=np.bool)
            seeds = np.array(seeds, dtype=np.int)[solvable]
            solvable = solvable[solvable]

        for level_sampler in level_samplers:
            level_sampler.observe_external_unseen_sample(seeds, solvable)

    def _reconcile_level_store_and_samplers(self):
        """ wenjun:
        this calls LevelStore() class,
        and remove num_proc of seeds with minimum support
        """
        all_replay_seeds = set()
        for level_sampler in self.all_level_samplers:
            all_replay_seeds.update([x for x in level_sampler.seeds if x >= 0])
        self.level_store.reconcile_seeds(all_replay_seeds)

    def _get_weighted_num_edits(self):
        level_sampler = self.all_level_samplers[0]
        seed_num_edits = np.zeros(level_sampler.seed_buffer_size)
        for idx, value in enumerate(self.level_store.seed2parent.values()):
            seed_num_edits[idx] = len(value)
        weighted_num_edits = np.dot(level_sampler.sample_weights(), seed_num_edits)
        return weighted_num_edits

    def _sample_replay_decision(self):
        return self._default_level_sampler.sample_replay_decision()

    def agent_rollout(self, 
                      agent, 
                      num_steps, 
                      update=False, 
                      is_env=False, 
                      level_replay=False, 
                      level_sampler=None, 
                      update_level_sampler=False,
                      discard_grad=False, 
                      edit_level=False,
                      num_edits=0, 
                      fixed_seeds=None):
        args = self.args
        if is_env:
            if edit_level: # Get mutated levels
                levels = [self.level_store.get_level(seed) for seed in fixed_seeds]
                self.ued_venv.reset_to_level_batch(levels)
                self.ued_venv.mutate_level(num_edits=num_edits)
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                return
            if level_replay: # Get replay levels
                # 1. sample replay levels according to probability, traj from a mixing envs. 
                # W/ Pc: Pc is needed for this sampling method
                self.current_level_seeds = [level_sampler.sample_replay_level() for _ in range(args.num_processes)]

                # 2. replay all levels in the buffer, traj from unique envs. 
                # W/O Pc: actually w/ or w/o Pc does not matter
                # self.current_level_seeds = level_sampler.seeds.tolist()

                # 3. replay the env with highest {replay prob} OR {regret}, traj from only one env. 
                # W/O Pc: pick the env only according to highest {replay prob} OR {regret}
                # self.current_level_seeds = [level_sampler.sample_highest_prob_level() for _ in range(args.num_processes)]

                levels = [self.level_store.get_level(seed) for seed in self.current_level_seeds]
                self.ued_venv.reset_to_level_batch(levels)
                self.replay_seeds_this_epi = self.current_level_seeds
                
                return self.current_level_seeds
            elif self.is_dr and not args.use_plr: 
                # print('1. use plr+ to dr generate envs...')
                obs = self.ued_venv.reset_random() # don't need obs here
                self.total_seeds_collected += args.num_processes
                return
            elif self.is_dr and args.use_plr and args.use_reset_random_dr:
                # print('2. use plr+ to dr generate envs...')
                obs = self.ued_venv.reset_random() # don't need obs here
                self._update_plr_with_current_unseen_levels(parent_seeds=fixed_seeds)
                self.total_seeds_collected += args.num_processes
                return
            elif self.is_alp_gmm:
                obs = self.alp_gmm_teacher.set_env_params(self.ued_venv)
                self.total_seeds_collected += args.num_processes
                return
            else:
                # print('3. use env_adv to generate envs...')       # this is actually the default generation. Use a uniformly random env_adv to create envs
                obs = self.ued_venv.reset() # Prepare for constructive rollout
                self.total_seeds_collected += args.num_processes
                self.replay_seeds_this_epi = [-1]
        else:
            obs = self.venv.reset_agent()


        """ student epi: update env_params, and compare it to env_params_in_buffer, and compute diversity """
        '''
        if not is_env:
            self.env_params = torch.clone(self._get_active_levels_params_vec())
            # print('epi-{}: 2. student epi, compute diversity, input_level={}, \n buffer_level={}'.format(self.current_epi, self.env_params[0:3], self.env_params_in_buffer[0:3]))
        '''

        # Initialize first observation
        agent.storage.copy_obs_to_index(obs,0)
        mean_return = 0

        rollout_returns = [[] for _ in range(args.num_processes)]
        for step in range(num_steps):
            
            if args.render:
                self.venv.render_to_screen()
            # Sample actions
            with torch.no_grad():
                obs_id = agent.storage.get_obs(step)
                value, action, action_log_dist, recurrent_hidden_states = agent.act(obs_id, agent.storage.get_recurrent_hidden_state(step), agent.storage.masks[step])
                if self.is_discrete_actions:
                    action_log_prob = action_log_dist.gather(-1, action)
                else:
                    action_log_prob = action_log_dist

            # Observe reward and next obs
            reset_random = self.is_dr and not args.use_plr
            _action = agent.process_action(action.cpu())

            if is_env:
                obs, reward, done, infos = self.ued_venv.step_adversary(_action)
            else:
                obs, reward, done, infos = self.venv.step_env(_action, reset_random=reset_random)
                if args.clip_reward:
                    reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)

            if not is_env and step >= num_steps - 1:
                # Handle early termination due to cliffhanger rollout
                if agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated'] = True
                            infos[i]['truncated_obs'] = get_obs_at_index(obs, i)

                done = np.ones_like(done, dtype=np.float)

            if level_sampler and level_replay:
                next_level_seeds = [s for s in self.current_level_seeds]
                
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    rollout_returns[i].append(info['episode']['r'])

                    if reset_random:
                        self.total_seeds_collected += 1

                    if not is_env:
                        self.total_episodes_collected += 1

                        # Handle early termination
                        if agent.storage.use_proper_time_limits:
                            if 'truncated_obs' in info.keys():
                                truncated_obs = info['truncated_obs']
                                agent.storage.insert_truncated_obs(truncated_obs, index=i)

                        # If using PLR, sample next level
                        if level_sampler and level_replay:
                            level_seed = level_sampler.sample_replay_level()
                            level = self.level_store.get_level(level_seed)
                            obs_i = self.venv.reset_to_level(level, i)
                            set_obs_at_index(obs, obs_i, i)
                            next_level_seeds[i] = level_seed
                            self.current_level_seeds[i] = level_seed

                        # If using ALP-GMM, sample next level
                        if self.is_alp_gmm:
                            self.alp_gmm_teacher.record_train_episode(rollout_returns[i][-1], index=i)
                            self.alp_gmm_teacher.set_env_params(self.venv)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'truncated' in info.keys() else [1.0] for info in infos])
            cliffhanger_masks = torch.FloatTensor([[0.0] if 'cliffhanger' in info.keys() else [1.0] for info in infos])

            # Need to store level seeds alongside non-env agent steps
            current_level_seeds = None
            if (not is_env) and level_sampler:
                current_level_seeds = torch.tensor(self.current_level_seeds, dtype=torch.int).view(-1, 1)
            
            agent.insert(obs, recurrent_hidden_states, action, action_log_prob, 
                         action_log_dist, value, reward, masks, bad_masks, 
                         level_seeds=current_level_seeds,
                         cliffhanger_masks=cliffhanger_masks)
            
            if level_sampler and level_replay:
                self.current_level_seeds = next_level_seeds


            # =========================================================== # 
            #              compute Euclidean(traj) here
            """
            1. student obs space: minigrid: obs={ image=(32,3,5,5), direction=(32,1)}, range={image=[], direction={0,1,2,3} }
                                  bipedal:  obs={ image=(32,24) }
            
            2. obs is not normalized for bipedal, check dim range at https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
            
        
            3. wenjun: the last step obs is all 0, i.e. obs[2049]=[0,0,0,...,0]. So, remove it, now it's (1,2048,24)

            """

            # wenjun: store (s,a)~new_traj and (s,a)~buffer_traj
            if not is_env and level_replay:
                obs_normed = obs / self.OBS_NORMALIZE
                # self.Buffer_Trajs[step] = torch.clone(obs_normed).cuda()
                self.Buffer_Trajs[step] = obs_normed
            if not is_env and not level_replay:
                obs_normed = obs / self.OBS_NORMALIZE
                # self.New_Trajs[step] = torch.clone(obs_normed).cuda()
                self.New_Trajs[step] = obs_normed
        
        # when student rollout on new levels is done, compute divergent for {new_trajs, buffer_trajs}
        if not is_env and level_replay:
            self.ever_replayed = True
        
        # print('epi-{}, agent={}, level_replay={}, ever_replayed={}'.format(self.current_epi, 'env_adv' if is_env else 'student', level_replay, self.ever_replayed))


        # if not is_env and not level_replay and self.ever_replayed:
        if not is_env and self.ever_replayed:                           # wenjun: also need to compute div for each level within the buffer
            for new_idx in range(self.args.num_processes):
                # 1. div_def = 'avg', sum over all div(tau, Tau)
                '''
                if self.div_definition == 'avg':
                    div_idx = 0
                    for buffer_idx in range(self.args.num_processes):
                        new_traj = self.New_Trajs[:-1, new_idx, :]
                        buffer_traj = self.Buffer_Trajs[:-1, buffer_idx, :]
                        
                        # try different losses
                        # loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)    # sinkhorn
                        loss = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur,)    
                        
                        with torch.no_grad():
                            div = loss(new_traj, buffer_traj)  
                        div_idx += div   
                '''
                # 2. div_def = 'min', take the min of div(tau, Tau)
                # elif self.div_definition == 'min':
                div_list = []
                for buffer_idx in range(self.args.num_processes):
                    if new_idx == buffer_idx:
                        # cannot compare to self
                        pass
                    else:
                        if level_replay:
                            new_traj = self.Buffer_Trajs[:-1, new_idx, :]
                            buffer_traj = self.Buffer_Trajs[:-1, buffer_idx, :]
                        else:
                            new_traj = self.New_Trajs[:-1, new_idx, :]
                            buffer_traj = self.Buffer_Trajs[:-1, buffer_idx, :]
                    
                        # Cost = Euclidean 
                        loss = SamplesLoss(loss="sinkhorn", p=self.p, blur=self.blur,)    
                        with torch.no_grad():
                            div = loss(new_traj, buffer_traj)
                        div_list.append(div) 
                div_idx = min(div_list)
                
                # print('div_idx = {}, div_list={}'.format(div_idx, div_list))
                self.level_divgence[new_idx] = div_idx

            agent.insert_divergence(level_divergence=self.level_divgence)
            
            # log sum of div within the buffer
            if level_replay:
                self.divergence_sum = torch.clone(torch.sum(self.level_divgence))
        # =========================================================== # 



        # Add generated env to level store (as a constructive string representation)
        if is_env and args.use_plr and not level_replay:
            self._update_plr_with_current_unseen_levels()

        rollout_info = self._get_rollout_return_stats(rollout_returns)

        # Update non-env agent if required
        if not is_env and update: 
            with torch.no_grad():
                obs_id = agent.storage.get_obs(-1)
                next_value = agent.get_value(
                    obs_id, agent.storage.get_recurrent_hidden_state(-1),
                    agent.storage.masks[-1]).detach()

            agent.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)

            # Compute batched value loss if using value_l1-maximizing adversary
            if self.requires_batched_vloss:
                # Don't clip value loss reward if env adversary normalizes returns
                clipped = not args.adv_use_popart and not args.adv_normalize_returns
                batched_value_loss = self._get_batched_value_loss(agent, clipped=clipped, batched=True)
                rollout_info.update({'batched_value_loss': batched_value_loss})

            # Update level sampler and remove any ejected seeds from level store
            if level_sampler and update_level_sampler:
                """
                # wenjun: update level_sampler with diversity metric                
                """
                level_sampler.update_with_rollouts(agent.storage)

            value_loss, action_loss, dist_entropy, info = agent.update(discard_grad=discard_grad)

            if level_sampler and update_level_sampler:
                level_sampler.after_update()

            rollout_info.update({
                'value_loss': value_loss,
                'action_loss': action_loss,
                'dist_entropy': dist_entropy,
                'update_info': info,
            })

            # Compute LZ complexity of action trajectories
            if args.log_action_complexity:
                rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})
        
        return rollout_info

    def _compute_env_return(self, agent_info, adversary_agent_info):
        args = self.args
        if args.ued_algo == 'paired':
            env_return = torch.max(adversary_agent_info['max_return'] - agent_info['mean_return'], \
                torch.zeros_like(agent_info['mean_return']))

        elif args.ued_algo == 'flexible_paired':
            env_return = torch.zeros_like(agent_info['max_return'], dtype=torch.float, device=self.device)
            adversary_agent_max_idx = adversary_agent_info['max_return'] > agent_info['max_return']
            agent_max_idx = ~adversary_agent_max_idx

            env_return[adversary_agent_max_idx] = \
                adversary_agent_info['max_return'][adversary_agent_max_idx]
            env_return[agent_max_idx] = agent_info['max_return'][agent_max_idx]
            
            env_mean_return = torch.zeros_like(env_return, dtype=torch.float)
            env_mean_return[adversary_agent_max_idx] = \
                agent_info['mean_return'][adversary_agent_max_idx]
            env_mean_return[agent_max_idx] = \
                adversary_agent_info['mean_return'][agent_max_idx]

            env_return = torch.max(env_return - env_mean_return, torch.zeros_like(env_return))

        elif args.ued_algo == 'minimax':
            env_return = -agent_info['max_return']

        else:
            env_return = torch.zeros_like(agent_info['mean_return'])

        if args.adv_normalize_returns:
            self.env_return_rms.update(env_return.flatten().cpu().numpy())
            env_return /= np.sqrt(self.env_return_rms.var + 1e-8)

        if args.adv_clip_reward is not None:
            clip_max_abs = args.adv_clip_reward
            env_return = env_return.clamp(-clip_max_abs, clip_max_abs)
        
        return env_return



    def run(self, epi):
        self.current_epi = epi
        args = self.args

        adversary_env = self.agents['adversary_env']
        agent = self.agents['agent']
        adversary_agent = self.agents['adversary_agent']

        level_replay = False
        if args.use_plr and self.is_training:
            level_replay = self._sample_replay_decision()

        # Discard student gradients if not level replay (sampling new levels)
        student_discard_grad = False
        no_exploratory_grad_updates = \
            vars(args).get('no_exploratory_grad_updates', False)
        if args.use_plr and (not level_replay) and no_exploratory_grad_updates:
            student_discard_grad = True

        if self.is_training and not student_discard_grad:
            self.student_grad_updates += 1

        print('level_replay', level_replay)
        print('buffer seeds', self._get_level_sampler('agent')[0].seeds)

        # Generate a batch of adversarial environments
        env_info = self.agent_rollout(
            agent=adversary_env, 
            num_steps=self.adversary_env_rollout_steps, 
            update=False,
            is_env=True,
            level_replay=level_replay,
            level_sampler=self._get_level_sampler('agent')[0],
            update_level_sampler=False)


        # Run agent episodes
        level_sampler, is_updateable = self._get_level_sampler('agent')
        agent_info = self.agent_rollout(
            agent=agent, 
            num_steps=self.agent_rollout_steps,
            update=self.is_training,
            level_replay=level_replay,
            level_sampler=level_sampler,
            update_level_sampler=is_updateable,
            discard_grad=student_discard_grad)
        
        print('first rollout')
        print('buffer seeds',self._get_level_sampler('agent')[0].seeds)

        # Use a separate PLR curriculum for the antagonist
        if level_replay and self.is_paired and (args.protagonist_plr == args.antagonist_plr):
            self.agent_rollout(
                agent=adversary_env, 
                num_steps=self.adversary_env_rollout_steps, 
                update=False,
                is_env=True,
                level_replay=level_replay,
                level_sampler=self._get_level_sampler('adversary_agent')[0],
                update_level_sampler=False)

        adversary_agent_info = defaultdict(float)
        if self.is_paired:
            # Run adversary agent episodes
            level_sampler, is_updateable = self._get_level_sampler('adversary_agent')
            adversary_agent_info = self.agent_rollout(
                agent=adversary_agent, 
                num_steps=self.agent_rollout_steps, 
                update=self.is_training,
                level_replay=level_replay,
                level_sampler=level_sampler,
                update_level_sampler=is_updateable,
                discard_grad=student_discard_grad)

        # Sample whether the decision to edit levels
        edit_level = self._should_edit_level() and level_replay

        if level_replay:
            sampled_level_info = {
                'level_replay': True,
                'num_edits': [len(self.level_store.seed2parent[x])+1 for x in env_info],
            }
        else:
            sampled_level_info = {
                'level_replay': False,
                'num_edits': [0 for _ in range(args.num_processes)]
            }

        # ==== This part performs ACCEL ====
        # If editing, mutate levels just replayed by PLR
        if level_replay and edit_level:
            # Choose base levels for mutation
            if self.base_levels == 'batch':
                fixed_seeds = env_info
            elif self.base_levels == 'easy':
                if args.num_processes >= 4:
                    # take top 4
                    easy = list(np.argsort((agent_info['mean_return'].detach().cpu().numpy() - agent_info['batched_value_loss'].detach().cpu().numpy()))[:4])
                    fixed_seeds = [env_info[x.item()] for x in easy] * int(args.num_processes/4)
                else:
                    # take top 1
                    easy = np.argmax((agent_info['mean_return'].detach().cpu().numpy() - agent_info['batched_value_loss'].detach().cpu().numpy()))
                    fixed_seeds = [env_info[easy]] * args.num_processes

            level_sampler, is_updateable = self._get_level_sampler('agent')

            # Edit selected levels
            self.agent_rollout(
                agent=None,
                num_steps=None,
                is_env=True,
                edit_level=True,
                num_edits=args.num_edits,
                fixed_seeds=fixed_seeds)

            self.total_num_edits += 1
            sampled_level_info['num_edits'] = [x+1 for x in sampled_level_info['num_edits']]

            # Evaluate edited levels
            agent_info_edited_level = self.agent_rollout(
                agent=agent,
                num_steps=self.agent_rollout_steps,
                update=self.is_training,
                level_replay=False,
                level_sampler=level_sampler,
                update_level_sampler=is_updateable,
                discard_grad=True)
            
            print('edited rollout')
            print('buffer seeds',self._get_level_sampler('agent')[0].seeds)
        # ==== ACCEL end ====

        if args.use_plr:
            self._reconcile_level_store_and_samplers()
            if self.use_editor:
                self.weighted_num_edits = self._get_weighted_num_edits()

        # Update adversary agent final return
        env_return = self._compute_env_return(agent_info, adversary_agent_info)


        # wenjun: check seeds after updated
        # print('epi-{}, level_replay={}, level_buffer seeds={}'.format(epi, level_replay, level_sampler.seeds))


        adversary_env_info = defaultdict(float)
        if self.is_training and self.is_training_env:
            with torch.no_grad():
                obs_id = adversary_env.storage.get_obs(-1)
                next_value = adversary_env.get_value(
                    obs_id, adversary_env.storage.get_recurrent_hidden_state(-1),
                    adversary_env.storage.masks[-1]).detach()
            adversary_env.storage.replace_final_return(env_return)
            adversary_env.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            env_value_loss, env_action_loss, env_dist_entropy, info = adversary_env.update()
            adversary_env_info.update({
                'action_loss': env_action_loss,
                'value_loss': env_value_loss,
                'dist_entropy': env_dist_entropy,
                'update_info': info
            })

        if self.is_training:
            self.num_updates += 1

        # === LOGGING ===
        # Only update env-related stats when run generates new envs (not level replay)
        log_replay_complexity = level_replay and args.log_replay_complexity
        if (not level_replay) or log_replay_complexity:
            stats = self._get_env_stats(agent_info, adversary_agent_info, log_replay_complexity=log_replay_complexity)
            stats.update({
                'mean_env_return': env_return.mean().item(),
                'adversary_env_pg_loss': adversary_env_info['action_loss'],
                'adversary_env_value_loss': adversary_env_info['value_loss'],
                'adversary_env_dist_entropy': adversary_env_info['dist_entropy'],
            })
            if args.use_plr:
                self.latest_env_stats.update(stats) # Log latest UED curriculum stats instead of PLR env stats
        else:
            stats = self.latest_env_stats.copy()

        # Log PLR buffer stats
        if args.use_plr and args.log_plr_buffer_stats:
            stats.update(self._get_plr_buffer_stats())

        [self.agent_returns.append(r) for b in agent_info['returns'] for r in reversed(b)]
        mean_agent_return = 0
        if len(self.agent_returns) > 0:
            mean_agent_return = np.mean(self.agent_returns)

        mean_adversary_agent_return = 0
        if self.is_paired:
            [self.adversary_agent_returns.append(r) for b in adversary_agent_info['returns'] for r in reversed(b)]
            if len(self.adversary_agent_returns) > 0:
                mean_adversary_agent_return = np.mean(self.adversary_agent_returns)

        self.sampled_level_info = sampled_level_info

        # wenjun: log replaybuffer passble ratio
        '''
        stats.update({
            'num_blocks_replaybuffer': self.replaybuffer_num_blocks,
            'passable_ratio_replaybuffer': self.replaybuffer_passable_ratio,
            'shortest_path_length_replaybuffer': self.replaybuffer_shortest_path_length,
        })
        '''

        #KELLY
        print('replay', level_replay)
        #what is the rate of entering buffer for edited and new seeds?
        candidate_seeds = self.current_level_seeds
        print('candidate_seeds', candidate_seeds)

        working_buffer_seeds = self.level_samplers["agent"].working_seed_set
        print('working_buffer_seeds', working_buffer_seeds)

        count_candidate_into_buffer = sum(el in candidate_seeds for el in working_buffer_seeds)
        print(count_candidate_into_buffer)

        if level_replay:
            stats.update({'num_edited_into_buffer': count_candidate_into_buffer})
            print('replay, num_edited')
        else:
            stats.update({'num_new_into_buffer': count_candidate_into_buffer})
            print('not replay, num_new')
            
        #average divergence of edited seeds -> but not just upon addition
        #also cuz diversity of new seeds not used for entry into buffer so its a bit useless
        #track in the buffer also cuz they might drop out
        # avg_divg_edited = self.divergence_sum/len(edited_seeds)
        # stats.update({'diversity_mean_edited': avg_divg_edited})
        
        #divergence scores for new seeds vs edited seeds upon entry to buffer
        print(self.level_divgence)
        level_divgence_sum = torch.clone(torch.sum(self.level_divgence))
        if level_replay:
            #last div buffer is filled with edited seeds
            stats.update({'divergence_sum_edited_seeds': level_divgence_sum})
            print('replay, edited div')
        else:
            stats.update({'divergence_sum_new_seeds': level_divgence_sum})
            print('not replay, new div')
        print('level_divgence_sum', level_divgence_sum)

        #what is regret of removed seeds, edited vs new
        removed_seed_edited_scores = []
        removed_seed_new_scores = []
        edited_seeds_removed = self.level_store.edited_seeds_removed
        print('edited_seeds_removed', edited_seeds_removed)

        edited_seeds_buffer_removed = set(edited_seeds_removed)-set(candidate_seeds)
        print('edited_seeds_buffer_removed', edited_seeds_buffer_removed)

        new_seeds_buffer_removed = self.level_store.ejected_seeds - set(edited_seeds_removed)-set(candidate_seeds)
        print('new_seeds_buffer_removed', new_seeds_buffer_removed)

        removed_seed_score_dict = level_sampler.removed_seed
        print('removed_seed_score_dict', removed_seed_score_dict)

        for removed_seed in edited_seeds_buffer_removed:
            removed_seed_edited_scores.append(removed_seed_score_dict[removed_seed])

        for removed_seed in new_seeds_buffer_removed:
            removed_seed_new_scores.append(removed_seed_score_dict[removed_seed])

        print('removed_seed_edited_scores',removed_seed_edited_scores)
        print('removed_seed_new_scores', removed_seed_new_scores)

        if len(edited_seeds_buffer_removed)!=0:
            removed_seed_edited_score = sum(removed_seed_edited_scores)/len(removed_seed_edited_scores)
            stats.update({'removed_seed_edited_mean_regret':removed_seed_edited_score})

            print('removed_seed_edited_mean_regret', removed_seed_edited_score)
        if len(new_seeds_buffer_removed)!=0:
            removed_seed_new_score = sum(removed_seed_new_scores)/len(removed_seed_new_scores)
            stats.update({'removed_seed_new_mean_regret':removed_seed_new_score})

            print('removed_seed_new_mean_regret', removed_seed_new_score)

        #number of dropped seeds (edited vs new)
        stats.update({'num_edited_seeds_removed':len(edited_seeds_buffer_removed),
                      'num_new_seeds_removed':len(new_seeds_buffer_removed)})
        print('edited_seeds_buffer_removed', len(edited_seeds_buffer_removed))
        print('num_new_seeds_removed', len(new_seeds_buffer_removed))

        # seeds_ejected_from_buffer = self.level_store.ejected_seeds - set(candidate_seeds)
        # if len(seeds_ejected_from_buffer)!=0:
        #     stats.update({'prop_edited_seeds_removed':len(edited_seeds_buffer_removed)/len(seeds_ejected_from_buffer)})
        

        # print('edited_seeds_removed', edited_seeds_removed)
        # print('level sampler removed', level_sampler.removed_seed)
        # for removed_seed in level_sampler.removed_seed:
        #     #seed itself -> check if edited
        #     if removed_seed[0] in edited_seeds_removed:
        #         #if edited seed
        #         print('edited seed removed', removed_seed[0], removed_seed[1])
        #         removed_seed_edited_scores.append(removed_seed[1])
        #     else:
        #         print('new seed removed', removed_seed[0], removed_seed[1])
        #         new_seed_edited_scores.append(removed_seed[1])


        #track droppage of edited seeds at point in time in level store (at removal from level store - _reconcile_level_store_and_samplers)
        # stats.update({'num_edited_seeds_removed':len(removed_seed_edited_scores)})
        # if self.level_store.num_removed_seeds!=0:
        #     stats.update({'prop_edited_seeds_removed':len(removed_seed_edited_scores)/(len(removed_seed_edited_scores)+len(new_seed_edited_scores))})
        # else:
        #     #give default of 0- not dropped
        #     stats.update({'prop_edited_seeds_removed':0})

        
        
        # KELLY: log scores for edited and new seeds (combined)
        # removed_seed_edited_scores = []
        # new_seed_edited_scores = []
        # edited_seeds_removed = self.level_store.edited_seeds_removed
        # for removed_seed in level_sampler.removed_seed:
        #     #seed itself -> check if edited
        #     if removed_seed[0] in edited_seeds_removed:
        #         #if edited seed
        #         removed_seed_edited_scores.append(removed_seed[1])
        #     else:
        #         new_seed_edited_scores.append(removed_seed[1])
        # if len(removed_seed_edited_scores)!=0:
        #     removed_seed_edited_score = sum(removed_seed_edited_scores)/len(removed_seed_edited_scores)
        #     stats.update({'removed_seed_edited_mean_score':removed_seed_edited_score})
        # if len(new_seed_edited_scores)!=0:
        #     new_seed_edited_score = sum(new_seed_edited_scores)/len(new_seed_edited_scores)
        #     stats.update({'new_seed_edited_mean_score':new_seed_edited_score})
        
        # # KELLY: log scores for seeds (regret) -> sum regret is alr done
        # level_sampler.seed_scores
        # # KELLY: log scores for seeds (diversity)
        # level_sampler.seed_diversity
        # level_sampler.seeds
        # idx_edited = np.isin(level_sampler.seeds, self.level_store.edited_seeds)
        # edited_diversity = np.mean(level_sampler.seed_diversity[idx_edited])
        # np.mean(level_sampler.seed_diversity[(~idx_edited) & (level_sampler.unseen_seed_weights<=0)])
        # #replay -> div sum (new) edited levels diversity which may not make it into buffer
        # #not replay -> ()

        #update average score of edited seeds and new seeds
        
        stats.update({
            'steps': (self.num_updates + self.total_num_edits) * args.num_processes * args.num_steps,
            'total_episodes': self.total_episodes_collected,
            'total_seeds': self.total_seeds_collected,
            'total_student_grad_updates': self.student_grad_updates,

            # log diversity metric
            'divergence_sum': self.divergence_sum,
            # 'diversity_sum': self.level_diversity_sum,
            'regret_sum': level_sampler.regret_sum,
            #KELLY log mean diversity for all seeds current working buffer
            # 'diversity_mean_buffer': level_sampler.diversity_mean,

            'mean_agent_return': mean_agent_return,
            'agent_value_loss': agent_info['value_loss'],
            'agent_pg_loss': agent_info['action_loss'],
            'agent_dist_entropy': agent_info['dist_entropy'],

            'mean_adversary_agent_return': mean_adversary_agent_return,
            'adversary_value_loss': adversary_agent_info['value_loss'],
            'adversary_pg_loss': adversary_agent_info['action_loss'],
            'adversary_dist_entropy': adversary_agent_info['dist_entropy'],
        })

        if args.log_grad_norm:
            agent_grad_norm = np.mean(agent_info['update_info']['grad_norms'])
            adversary_grad_norm = 0
            adversary_env_grad_norm = 0
            if self.is_paired:
                adversary_grad_norm = np.mean(adversary_agent_info['update_info']['grad_norms'])
            if self.is_training_env:
                adversary_env_grad_norm = np.mean(adversary_env_info['update_info']['grad_norms'])
            stats.update({
                'agent_grad_norm': agent_grad_norm,
                'adversary_grad_norm': adversary_grad_norm,
                'adversary_env_grad_norm': adversary_env_grad_norm
            })

        if args.log_action_complexity:
            stats.update({
                'agent_action_complexity': agent_info['action_complexity'],
                'adversary_action_complexity': adversary_agent_info['action_complexity']  
            }) 

        return stats
