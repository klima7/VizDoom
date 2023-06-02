import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule
import vizdoom as vzd

from .networks import Actor, Critic
from ..agent import Agent
from ..replay import ReplayBuffer, ReplayDataset


class ActorCriticAgent(LightningModule, Agent):


    def __init__(
            self,
            n_actions,
            screen_size,
            n_variables,
            batch_size=32,
            lr_actor=0.001,
            lr_critic=0.001,
            frames_skip=1,
            gamma=0.99,
            buffer_size=50_000,
            populate_steps=1_000,
            actions_per_step=10,
            validation_interval=50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.actor = Actor(self.hparams.n_actions, self.hparams.screen_size, self.hparams.n_variables)
        self.critic = Critic(self.hparams.screen_size, self.hparams.n_variables)

        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = None

        self.env = None
        self.train_metrics = {}
        self.val_metrics = {}

    def set_train_environment(self, env):
        self.env = env
        self.dataset = ReplayDataset(self.buffer, self.env)

    def get_action(self, state):
        with torch.no_grad():
            # print(state)
            action_probs = self.actor.forward_state(state, self.device)
            action_probs = action_probs.detach().cpu().numpy()
            action_idx = np.random.choice(self.hparams.n_actions, p=action_probs)
            action_vec = self.__get_action_vec(action_idx)
            return action_vec

    def configure_optimizers(self):
        optimizer_actor = Adam(self.actor.parameters(), lr=self.hparams.lr_actor)
        optimizer_critic = Adam(self.critic.parameters(), lr=self.hparams.lr_critic)
        return optimizer_actor, optimizer_critic

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def on_fit_start(self):
        self.env.init()
        self.__populate_buffer()

    def on_fit_end(self):
        self.env.close()

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.validation_interval == 0:
            self.__validate()

    def on_train_batch_start(self, batch, batch_idx):
        for _ in range(self.hparams.actions_per_step):
            done = self.__play_step(update_buffer=True)
            if done:
                self.train_metrics = self.env.get_metrics(prefix='train_')
                break

    def training_step(self, batch, batch_no):
        states, actions, rewards, next_states, dones = batch
        actor_optimizer, critic_optimizer = self.optimizers()

        td_error = self.__compute_td_error(states, actions, rewards, next_states, dones)

        actor_optimizer.zero_grad()
        actor_loss = -torch.log(self.actor(states).gather(1, actions.unsqueeze(1)).squeeze()) * td_error.detach()
        actor_loss = actor_loss.mean()
        self.manual_backward(actor_loss)
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss = td_error.pow(2)
        critic_loss = critic_loss.mean()
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        self.log('actor_loss', actor_loss),
        self.log('critic_loss', critic_loss),
        self.log('train_buffer_size', float(len(self.buffer)))
        self.log_dict(self.train_metrics)
        self.log_dict(self.val_metrics)

    def __compute_td_error(self, states, actions, rewards, next_states, dones):
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_targets = rewards + self.hparams.gamma * next_values * (1 - dones.long())
        td_error = td_targets - values
        return td_error

    def __play_step(self, update_buffer=True):
        state = self.env.get_state()
        action = self.get_action(state)

        try:
            self.env.make_action(action, skip=self.hparams.frames_skip)
        except (vzd.vizdoom.SignalException, vzd.vizdoom.ViZDoomUnexpectedExitException):
            raise KeyboardInterrupt

        done = self.env.is_episode_finished()

        if update_buffer:
            reward = self.env.get_last_reward()
            next_state = self.env.get_state()
            self.buffer.add(state, np.argmax(action), reward, next_state, done)

        return done

    def __populate_buffer(self):
        while len(self.buffer) < self.hparams.populate_steps:
            done = self.__play_step()
            if done:
                self.env.new_episode()

    def __validate(self):
        self.env.new_episode()
        while not self.env.is_episode_finished():
            state = self.env.get_state()
            action = self.get_action(state)
            self.env.make_action(action)
        self.val_metrics = self.env.get_metrics(prefix='val_')

    def __get_action_vec(self, action_idx):
        action_vector = [0] * self.hparams.n_actions
        action_vector[action_idx] = 1
        return action_vector
