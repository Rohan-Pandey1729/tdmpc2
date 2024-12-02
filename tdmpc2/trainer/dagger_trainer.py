from copy import deepcopy

from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from tdmpc2 import TDMPC2
from common.buffer import Buffer
from torch import Tensor
from typing import Union

# cfg.eval_episodes: Number of episodes to plot for evaluation.
# cfg.save_video: Whether or not to save the videos of the evaluations.
# cfg.dagger_epochs: How many times do we sample expert data + train model.
# cfg.trajs_per_dagger_epoch: How many trajectories do we add per epoch.
# cfg.train_epochs: How many times do we go through the buffer per dagger epoch.
class DaggerTrainer(Trainer):
  def __init__(self, cfg, env: Union[PixelWrapper, MultitaskWrapper, TensorWrapper], agent: TDMPC2, buffer: Buffer, logger):
    super().__init__(cfg, env, agent, buffer, logger)
    # In order to train the agent policy, we should keep the expert planning consistent.
    # This means we need to keep a constant expert model and policy by doing a deep copy.
    self.expert = deepcopy(agent)
    # Set student's mpc (planning) to false, because we don't want the student to plan.
    self.agent.cfg.mpc = False
  
  def eval(self):
    """Evaluate a TD-MPC2 agent."""
    ep_rewards, ep_successes = [], []

    # copied code from OnlineTrainer.
    for i in range(self.cfg.eval_episodes):
      obs, done, ep_reward, t = self.env.reset(), False, 0, 0
      if self.cfg.save_video:
        self.logger.video.init(self.env, enabled=(i==0))
      while not done:
        torch.compiler.cudagraph_mark_step_begin()
        action = self.agent.act(obs, t0=t==0, eval_mode=True)
        obs, reward, done, info = self.env.step(action)
        ep_reward += reward
        t += 1
        if self.cfg.save_video:
          self.logger.video.record(self.env)
      ep_rewards.append(ep_reward)
      ep_successes.append(info['success'])
      if self.cfg.save_video:
        self.logger.video.save(self._step)

    # return average reward and success rate.
    return dict(
      episode_reward=np.nanmean(ep_rewards),
      episode_success=np.nanmean(ep_successes),
    )

  def train(self):
    """Train a TD-MPC2 agent."""
    # Assume there's already a well-trained world model loaded.
    for dagger_i in range(self.cfg.dagger_epochs):
      # Rollout student policy and label with expert action.
      # Assume there's no buffer at the beginning.
      for traj_i in range(self.cfg.trajs_per_dagger_epoch):
        obs, done, t = self.env.reset(), False, 0
        while not done:
          student_action = self.agent.act(obs, t == 0, False)
          expert_action = self.expert.act(obs, t == 0, False) # Contrary to eval, here we keep eval_mode False.
          td = self.to_td(obs, expert_action, None) # We don't need reward in the buffer.
          self.buffer.add(td)
          obs, reward, done, info = self.env.step(student_action)
    
      # Train student policy with expert action.
      for train_i in range(self.cfg.train_epochs):
        obs, expert_action, reward, task = self.buffer.sample()
        obs_z = self.agent.model.encode(obs, task) # I'm kind of perplexed in this respect. We're not adding task into buffer, so how is it getting task?
        log_probs = self.agent.model.log_prob(obs_z, task, expert_action) # Custom-implemented function.
        loss = -torch.mean(log_probs)
        self.agent.optim.zero_grad()
        loss.backward()
        self.agent.optim.step()
  
  
  # Copied from OnlineTrainer.
  def to_td(self, obs: Union[dict, Tensor], action: Union[Tensor, None] = None, reward: Union[Tensor, None] = None):
    """Creates a TensorDict for a new episode."""
    if isinstance(obs, dict):
      obs = TensorDict(obs, batch_size=(), device='cpu')
    else:
      obs = obs.unsqueeze(0).cpu()
    if action is None:
      action = torch.full_like(self.env.rand_act(), float('nan'))
    if reward is None:
      reward = torch.tensor(float('nan'))
    td = TensorDict(
      obs=obs,
      action=action.unsqueeze(0),
      reward=reward.unsqueeze(0),
      batch_size=(1,)
    )
    return td
