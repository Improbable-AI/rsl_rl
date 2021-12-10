# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from ml_logger import logger

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from params_proto.neo_proto import PrefixProto

class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0
    wall_time = 0
    frame = 0


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None and self.cfg["use_tensorboard"]: # tensorboard
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        elif not self.cfg["use_tensorboard"]: # ml_logger
            #logger.configure('/tmp/ml-logger-debug')
            input(logger.prefix)

            assert logger.prefix, "you will overwrite the entire instrument server"
            if logger.read_params('job.completionTime', default=None):
                logger.print("The job seems to have been already completed!!!")
                return
            logger.start('update_job_status')
            logger.start('start', 'episode', 'run', 'step')

            if logger.glob('progress.pkl'):
                try:
                    # Use current config for some args
                    keep_args = ['checkpoint_root', 'time_limit', 'checkpoint_freq', 'tmp_dir']
                    Args._update({key: val for key, val in logger.read_params("Args").items() if key not in keep_args})
                except KeyError as e:
                    print('Captured KeyError during Args update.', e)

                agent, replay_buffer, progress_cache = load_checkpoint()
                Progress._update(progress_cache)
                logger.timer_cache['start'] = logger.timer_cache['start'] - Progress.wall_time
                logger.print(f'loaded from checkpoint at {Progress.episode}', color="green")

            else:
                Args._update(kwargs)
                logger.log_params(Args=vars(Args))
                logger.log_text("""
                    charts:
                    - yKey: train/episode_reward/mean
                      xKey: step
                    - yKey: eval/episode_reward
                      xKey: step
                    """, filename=".charts.yml", dedent=True, overwrite=True)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        if self.cfg["use_tensorboard"]:
            ep_string = self.log_tensorboard(locs, width=width, pad=pad)
        else:
            ep_string = self.log_ml_logger(locs, width=width, pad=pad)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def log_tensorboard(self, locs, width=80, pad=35):
        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        return ep_string

    def log_ml_logger(self, locs, width=80, pad=35):
        print("logging with ml-logger")

        log_dict = {}
        log_dict['it'] = locs['it']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                log_dict['Episode/' + key] =  value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        log_dict['Loss/value_function'] = locs['mean_value_loss']
        log_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        log_dict['Loss/learning_rate'] = self.alg.learning_rate
        log_dict['Policy/mean_noise_std'] = mean_std.item()
        log_dict['Perf/total_fps'] = fps
        log_dict['Perf/collection time'] = locs['collection_time']
        log_dict['Perf/learning_time'] = locs['learn_time']
        
        if len(locs['rewbuffer']) > 0:
            log_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            log_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            log_dict['Train/mean_reward/time'] = statistics.mean(locs['rewbuffer'])
            log_dict['Train/mean_episode_length/time'] = statistics.mean(locs['lenbuffer'])
            log_dict['tot_time'] = self.tot_time

        logger.store_metrics(log_dict)

        return ep_string


    def save(self, path, infos=None):
        
        dict_to_save = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }

        if self.cfg.ml_logger:
            #from .config import Args
            logger.pring('saving & uploading snapshot...')
            replay_path = pJoin(Args.checkpoint_root, logger.prefix, 'replay.pt')
            snapshot_target = pJoin(Args.checkpoint_root, logger.prefix, 'snapshot_in_progress.pt')
            logger.save_torch(dict_to_save, path=snapshot_target)

            logger.duplicate("metrics.pkl", "metrics_latest.pkl")
            logger.print('saving buffer to', replay_path)

            # NOTE: It seems tempfile cannot handle this 20GB+ file.
            logger.start('upload_replay_buffer')

            if replay_path.startswith('file://'):
                replay_path = replay_path[7:]
                Path(replay_path).resolve().parents[0].mkdir(parents=True, exist_ok=True)
                with open(replay_path, 'wb') as f:
                    cloudpickle.dump(replay, f)
            elif replay_path.startswith('s3://') or replay_path.startswith('gs://'):
                logger.print('uploading buffer to', replay_path)
                tmp_path = Path(Args.tmp_dir) / logger.prefix / 'replay.pt'
                tmp_path.parents[0].mkdir(parents=True, exist_ok=True)
                with open(tmp_path, 'wb') as f:
                    cloudpickle.dump(replay, f)

                if replay_path.startswith('s3://'):
                    logger.upload_s3(str(tmp_path), path=replay_path[5:])
                else:
                    logger.upload_gs(str(tmp_path), path=replay_path[5:])
            else:
                ValueError('replay_path must start with s3://, gs:// or file://. Not', replay_path)

            elapsed = logger.since('upload_replay_buffer')
            logger.print(f'Uploading replay buffer took {elapsed} seconds')

            # Save the progress.pkl last as a fail-safe. To make sure the checkpoints are saving correctly.
            logger.log_params(Progress=vars(Progress), path="progress.pkl", silent=True)

        else:
            torch.save(dict_to_save, path)

    def load(self, path, load_optimizer=True):
        if self.cfg.ml_logger:
            from .config import Args
            from ml_logger import logger
            import torch

            # TODO: check if both checkpoint & replay buffer exist
            snapshot_path = os.path.join(Args.checkpoint_root, logger.prefix, 'snapshot_in_progress.pt')
            replay_path = os.path.join(Args.checkpoint_root, logger.prefix, 'replay.pt')
            assert logger.glob(snapshot_path) and logger.glob(replay_path) and logger.glob('progress.pkl') and logger.glob('metrics_latest.pkl')

            logger.print('loading agent from', snapshot_path)
            loaded_dict = logger.load_torch(snapshot_path)

            # Load replay buffer
            logger.print('loading from checkpoint (replay)', replay_path)

            # NOTE: It seems tempfile cannot handle this 20GB+ file.
            logger.start('download_replay_buffer')
            if replay_path.startswith('file://'):
                import cloudpickle
                with open(replay_path[7:], 'rb') as f:
                    replay = cloudpickle.load(f)
            elif replay_path.startswith('s3://') or replay_path.startswith('gs://'):
                import cloudpickle
                tmp_path = Path(Args.tmp_dir) / logger.prefix / 'replay.pt'
                tmp_path.parents[0].mkdir(parents=True, exist_ok=True)

                if replay_path.startswith('s3://'):
                    logger.download_s3(path=replay_path[5:], to=str(tmp_path))
                else:
                    logger.download_gs(path=replay_path[5:], to=str(tmp_path))
                with open(tmp_path, 'rb') as f:
                    replay = cloudpickle.load(f)
            else:
                ValueError('replay_path must start with s3://, gs:// or file://. Not', replay_path)

            elapsed = logger.since('download_replay_buffer')
            logger.print(f'Download completed. It took {elapsed} seconds')

            logger.duplicate("metrics_latest.pkl", to="metrics.pkl")
            logger.print('done')

            params = logger.read_params(path="progress.pkl")

        else:
            loaded_dict = torch.load(path)


        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
