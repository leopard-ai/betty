from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

import gym

from betty.problems import ImplicitProblem
from betty.envs import Env

from data import ExperienceSourceDataset


class Actor(ImplicitProblem):
    def training_step(self, batch):
        state = batch["state"]
        action = batch["action"]
        logp_old = batch["logp_old"]
        adv = batch["adv"]

        pi, _ = self.module(state)
        logp = self.module.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()

        return loss_actor

    def get_batch(self):
        return self.env.batch


class Critic(ImplicitProblem):
    def training_step(self, batch):
        state = batch["state"]
        qval = batch["qval"]

        value = self.module(state)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def get_batch(self):
        return self.env.batch


class PPOEnv(Env):
    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lam: float = 0.95,
        max_episode_len: float = 1000,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
    ):
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio

        self.env = gym.make(env)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = torch.FloatTensor(self.env.reset())

        dataset = ExperienceSourceDataset(self.train_batch)
        self.train_data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size)

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(
        self, rewards: List[float], values: List[float], last_value: float
    ) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [
            rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)
        ]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def train_batch(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):
            if not hasattr(self, "state"):
                self.state = torch.FloatTensor(self.env.reset()).to(self.device)
            pi, action = self.actor(self.state)
            log_prob = self.actor.get_log_prob(pi, action)
            value = self.critic(self.state)

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        value = self.critic(self.state).to(self.device)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(
                    self.ep_rewards + [last_value], self.gamma
                )[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(
                    self.ep_rewards, self.ep_values, last_value
                )
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.FloatTensor(self.env.reset())

            if epoch_end:
                train_data = zip(
                    self.batch_states,
                    self.batch_actions,
                    self.batch_logp,
                    self.batch_qvals,
                    self.batch_adv,
                )

                for state, action, logp_old, qval, adv in train_data:
                    adv = (adv - adv.mean()) / adv.std()
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (
                    self.steps_per_epoch - steps_before_cutoff
                ) / nb_episodes

                self.epoch_rewards.clear()
