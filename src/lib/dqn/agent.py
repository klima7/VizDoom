import random

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import vizdoom as vzd

from .network import DQNNetwork
from ..agent import Agent
from ..replay import ReplayBuffer, ReplayDataset


class DQNAgent(LightningModule, Agent):

    def __init__(
            self,
            n_actions,
            lr=0.001,
            batch_size=32,
            frames_skip=1,
            epsilon=0.5,
            gamma=0.99,
            buffer_size=50_000,
            populate_steps=1_000,
            actions_per_step=10,
            weights_update_interval=1_000,
            epsilon_update_interval=200,
            epsilon_decay=0.99,
            epsilon_min=0.02,
            validation_interval=100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DQNNetwork(self.hparams.n_actions)
        self.target_model = DQNNetwork(self.hparams.n_actions)

        self.model.eval()
        self.target_model.train()

        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = ReplayDataset(self.buffer, self.hparams.batch_size)

        self.env = None
        self.train_metrics = {}

    def set_train_environment(self, env):
        self.env = env

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

    def __update_weights(self):
        self.model.load_state_dict(self.target_model.state_dict())

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
            self.buffer.add(state, action, reward, next_state, done)

        return done

    def __get_action_vec(self, action_idx):
        action_vector = [0] * self.hparams.n_actions
        action_vector[action_idx] = 1
        return action_vector

    def on_fit_start(self):
        self.env.init()
        self.__populate_buffer()

    def on_fit_end(self):
        self.env.close()

    # def on_train_epoch_end(self):
    #     if self.current_epoch % self.hparams.validation_interval == 0:
    #         self.__validate()
    #
    # def __validate(self):
    #     while not self.env.is_episode_finished():
    #         state = self.env.get_state()
    #         action = self.get_action(state, epsilon=0)
    #         self.env.make_action(action)
    #
    #     self.val_metrics = self.__get_metrics(prefix='val_')
    #     self.env.new_episode()

    def __populate_buffer(self):
        while len(self.buffer) < self.hparams.populate_steps:
            done = self.__play_step()
            if done:
                self.env.new_episode()
        self.env.new_episode()

    def training_step(self, batch, batch_no):
        for _ in range(self.hparams.actions_per_step):
            done = self.__play_step(update_buffer=True)

            if done:
                self.dataset.end_epoch()
                self.train_metrics = self.env.get_metrics(prefix='train_')
                self.env.new_episode()
                break

        loss = self.__calculate_loss(batch)

        if self.global_step % self.hparams.weights_update_interval == 0:
            self.__update_weights()

        if self.global_step % self.hparams.epsilon_update_interval == 0:
            self.hparams.epsilon = max(self.hparams.epsilon * self.hparams.epsilon_decay, self.hparams.epsilon_min)

        self.log('train_loss', loss),
        self.log('train_epsilon', self.hparams.epsilon),
        self.log('train_buffer_size', float(len(self.buffer)))
        self.log_dict(self.train_metrics)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.target_model.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def __calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = torch.argmax(actions, dim=1)

        state_action_values = self.target_model(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        return nn.MSELoss()(state_action_values, expected_state_action_values)
