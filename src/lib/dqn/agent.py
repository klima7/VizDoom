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
            skip=1,
            epsilon=0.5,
            gamma=0.99,
            buffer_size=50_000,
            populate_steps=1_000,
            actions_per_step=10,
            update_weights_interval=1_000,
            validation_interval=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DQNNetwork(self.hparams.n_actions)
        self.target_model = DQNNetwork(self.hparams.n_actions)

        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = ReplayDataset(self.buffer, self.hparams.batch_size)
        self.env = None

        # metrics
        self.train_metrics = {}
        self.val_metrics = {}

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
        self.model.eval()
        with torch.no_grad():
            qvalues = self.model.forward_state(state, self.device)
        self.model.train()

        best_action_idx = torch.argmax(qvalues).item()
        best_action_vec = self.__get_action_vec(best_action_idx)
        return best_action_vec

    def __update_weights(self):
        self.model.load_state_dict(self.target_model.state_dict())

    def __play_step(self):
        state = self.env.get_state()
        action = self.get_action(state)

        try:
            self.env.make_action(action, skip=self.hparams.skip)
        except (vzd.vizdoom.SignalException, vzd.vizdoom.ViZDoomUnexpectedExitException):
            raise KeyboardInterrupt

        reward = self.env.get_last_reward()
        next_state = self.env.get_state()
        done = self.env.is_episode_finished()
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

    def on_train_epoch_start(self):
        self.env.new_episode()

    def on_train_epoch_end(self):
        if self.current_epoch % self.hparams.validation_interval == 0:
            self.__validate()

    def __validate(self):
        self.env.new_episode()

        while not self.env.is_episode_finished():
            state = self.env.get_state()
            action = self.get_action(state, epsilon=0)
            self.env.make_action(action)

        self.val_metrics = self.__get_metrics(prefix='val_')

    def __populate_buffer(self):
        while len(self.buffer) < self.hparams.populate_steps:
            done = self.__play_step()
            if done:
                self.env.new_episode()

    def training_step(self, batch, batch_no):
        for _ in range(self.hparams.actions_per_step):
            done = self.__play_step()

            if done:
                self.dataset.end_epoch()
                self.train_metrics = self.__get_metrics(prefix='train_')
                break

        loss = self.__calculate_loss(batch)

        if self.global_step % self.hparams.update_weights_interval == 0:
            self.__update_weights()

        if self.global_step % 2_000 == 0:
            self.hparams.epsilon = max(self.hparams.epsilon * 0.99, 0.02)

        self.log('train_loss', loss),
        self.log('train_epsilon', self.hparams.epsilon),
        self.log('train_buffer_size', float(len(self.buffer)))
        self.log_dict(self.train_metrics)
        self.log_dict(self.val_metrics)
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

    def __get_metrics(self, prefix=''):
        return {
            f'{prefix}total_reward': float(self.env.get_total_reward()),
            f'{prefix}frags_count': float(self.env.get_frags_count()),
            f'{prefix}suicides_count': float(self.env.get_suicides_count()),
            f'{prefix}deaths_count': float(self.env.get_deaths_count()),
            f'{prefix}hits_made_count': float(self.env.get_hits_made_count()),
            f'{prefix}hits_taken_count': float(self.env.get_hits_taken_count()),
            f'{prefix}items_collected_count': float(self.env.get_items_collected_count()),
            f'{prefix}damage_make_count': float(self.env.get_damage_make_count()),
            f'{prefix}damage_taken_count': float(self.env.get_damage_taken_count()),
            f'{prefix}armor_gained_count': float(self.env.get_secrets_count()),
            f'{prefix}armor_lost_count': float(self.env.get_armor_gained_count()),
            f'{prefix}health_gained_count': float(self.env.get_armor_lost_count()),
            f'{prefix}health_lost_count': float(self.env.get_health_gained_count()),
            f'{prefix}death_tics_count': float(self.env.get_health_lost_count()),
            f'{prefix}attack_not_ready_tics': float(self.env.get_death_tics_count()),
        }
