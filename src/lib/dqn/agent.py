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
            epsilon=0.5,
            gamma=0.99,
            buffer_size=50_000,
            populate_steps=1_000,
            actions_per_step=10,
            update_weights_interval=1_000
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DQNNetwork(self.hparams.n_actions)
        self.target_model = DQNNetwork(self.hparams.n_actions)

        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = ReplayDataset(self.buffer, self.hparams.batch_size)
        self.env = None

        # metrics
        self.total_reward = 0.0
        self.frags_count = 0.0
        self.suicides_count = 0.0
        self.deaths_count = 0.0
        self.hits_made_count = 0.0
        self.hits_taken_count = 0.0
        self.items_collected_count = 0.0
        self.damage_make_count = 0.0
        self.damage_taken_count = 0.0
        self.armor_gained_count = 0.0
        self.armor_lost_count = 0.0
        self.health_gained_count = 0.0
        self.health_lost_count = 0.0
        self.death_tics_count = 0.0
        self.attack_not_ready_tics = 0.0

    def set_train_environment(self, env):
        self.env = env

    def get_action(self, state):
        if random.random() < self.hparams.epsilon:
            return self.__get_random_action()
        else:
            return self.__get_best_action(state)

    def __get_random_action(self):
        action_idx = random.randint(0, self.hparams.n_actions - 1)
        action_vec = self.__get_action_vec(action_idx)
        return action_vec

    def __get_best_action(self, state):
        qvalues = self.model.forward_state(state, self.device)
        best_action_idx = torch.argmax(qvalues).item()
        best_action_vec = self.__get_action_vec(best_action_idx)
        return best_action_vec

    def __update_weights(self):
        self.model.load_state_dict(self.target_model.state_dict())

    def __play_step(self):
        state = self.env.get_state()
        action = self.get_action(state)

        try:
            self.env.make_action(action)
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
                self.__update_metrics()
                self.env.new_episode()

        loss = self.__calculate_loss(batch)

        if self.global_step % self.hparams.update_weights_interval == 0:
            self.__update_weights()
            self.hparams.epsilon = max(self.hparams.epsilon * 0.99, 0.02)

        self.__log_metrics(loss)
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

    def __log_metrics(self, loss, prefix=''):
        self.log_dict({
            'loss': loss,
            f'{prefix}epsilon': self.hparams.epsilon,
            f'{prefix}buffer_size': float(len(self.buffer)),
            f'{prefix}total_reward': self.total_reward,
            f'{prefix}frags_count': self.frags_count,
            f'{prefix}suicides_count': self.suicides_count,
            f'{prefix}deaths_count': self.deaths_count,
            f'{prefix}hits_made_count': self.hits_made_count,
            f'{prefix}hits_taken_count': self.hits_taken_count,
            f'{prefix}damage_make_count': self.damage_make_count,
            f'{prefix}damage_taken_count': self.damage_taken_count,
            f'{prefix}death_tics_count': self.death_tics_count,
        })

    def __update_metrics(self):
        self.total_reward = float(self.env.get_total_reward())
        self.frags_count = float(self.env.get_frags_count())
        self.suicides_count = float(self.env.get_suicides_count())
        self.deaths_count = float(self.env.get_deaths_count())
        self.hits_made_count = float(self.env.get_hits_made_count())
        self.hits_taken_count = float(self.env.get_hits_taken_count())
        self.items_collected_count = float(self.env.get_items_collected_count())
        self.damage_make_count = float(self.env.get_damage_make_count())
        self.damage_taken_count = float(self.env.get_damage_taken_count())
        self.armor_gained_count = float(self.env.get_secrets_count())
        self.armor_lost_count = float(self.env.get_armor_gained_count())
        self.health_gained_count = float(self.env.get_armor_lost_count())
        self.health_lost_count = float(self.env.get_health_gained_count())
        self.death_tics_count = float(self.env.get_health_lost_count())
        self.attack_not_ready_tics = float(self.env.get_death_tics_count())
