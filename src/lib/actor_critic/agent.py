import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule
import vizdoom as vzd

from .network import DQNNetwork
from ..agent import Agent
from ..replay import ReplayBuffer, ReplayDataset


class ActorCritic(LightningModule, Agent):

    def __init__(
            self,
            n_actions,
            screen_size,
            n_variables,
            lr=0.001,
            batch_size=32,
            frames_skip=1,
            epsilon=0.5,
            gamma=0.99,
            buffer_size=50_000,
            populate_steps=1_000,
            actions_per_step=10,
            validation_interval=10,
            weights_update_interval=1_000,
            epsilon_update_interval=200,
            epsilon_decay=0.99,
            epsilon_min=0.02,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.target_model = DQNNetwork(self.hparams.n_actions, self.hparams.screen_size, self.hparams.n_variables)
        self.model = DQNNetwork(self.hparams.n_actions, self.hparams.screen_size, self.hparams.n_variables)

        self.target_model.eval()
        self.model.train()
        self.__update_weights()

        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = None

        self.env = None
        self.train_metrics = {}
        self.val_metrics = {}

    def set_train_environment(self, env):
        self.env = env
        self.dataset = ReplayDataset(self.buffer, self.env)

    def get_action(self, state, epsilon=None):
        if random.random() < (epsilon or self.hparams.epsilon):
            return self.__get_random_action()
        else:
            return self.__get_best_action(state)

    def __get_random_action(self):
        action_idx = random.randint(0, self.hparams.n_actions - 1)
        action_vec = self.__get_action_vec(action_idx)
        return action_vec

    def __get_best_action(self, state):
        with torch.no_grad():
            qvalues = self.model.forward_state(state, self.device)

        best_action_idx = torch.argmax(qvalues).item()
        best_action_vec = self.__get_action_vec(best_action_idx)
        return best_action_vec

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr, amsgrad=True)
        return optimizer

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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % self.hparams.weights_update_interval == 0:
            self.__update_weights()

        if self.global_step % self.hparams.epsilon_update_interval == 0:
            self.hparams.epsilon = max(self.hparams.epsilon * self.hparams.epsilon_decay, self.hparams.epsilon_min)

    def training_step(self, batch, batch_no):
        loss = self.__calculate_loss(batch)

        self.log('train_loss', loss),
        self.log('train_epsilon', self.hparams.epsilon),
        self.log('train_buffer_size', float(len(self.buffer)))
        self.log_dict(self.train_metrics)
        self.log_dict(self.val_metrics)
        return loss

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

    def __calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

    def __update_weights(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def __populate_buffer(self):
        while len(self.buffer) < self.hparams.populate_steps:
            done = self.__play_step()
            if done:
                self.env.new_episode()

    def __validate(self):
        self.env.new_episode()
        while not self.env.is_episode_finished():
            state = self.env.get_state()
            action = self.get_action(state, epsilon=0)
            self.env.make_action(action)
        self.val_metrics = self.env.get_metrics(prefix='val_')

    def __get_action_vec(self, action_idx):
        action_vector = [0] * self.hparams.n_actions
        action_vector[action_idx] = 1
        return action_vector
