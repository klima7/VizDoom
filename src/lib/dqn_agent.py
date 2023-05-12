import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchinfo import summary

from .agent import Agent
from .replay import ReplayBuffer, ReplayDataset


class PreprocessStateGameWrapper:
    
    def __init__(self, game):
        self.game = game
        
    def __getattr__(self, attr):
        if attr == 'game':
            return self.game
        return getattr(self.game, attr)
    
    def get_state(self):
        state = self.game.get_state()
        screen_buffer = state.screen_buffer  # 240x320
        screen_buffer = screen_buffer[np.newaxis, ...]
        return screen_buffer


class Network(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3456, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


class DQNAgent(LightningModule, Agent):
    
    def __init__(
        self, 
        n_actions,
        batch_size=32,
        epsilon=0.5,
        gamma=0.99,
        buffer_size=10**5, 
        populate_steps=1_000,
        actions_per_step=10,
        update_weights_interval=1_000
        ):
        super().__init__()
        self.save_hyperparameters()
        print(n_actions, self.hparams.n_actions)
        self.model = Network(self.hparams.n_actions)
        self.target_model = Network(self.hparams.n_actions)
        
        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.state = None
        self.env = None
        
        self.episode_reward = 0
        self.total_reward = 0
    
    def set_train_environment(self, env):
        self.env = env
        
    def get_action(self, state):
        if random.random() < self.hparams.epsilon:
            return self.__get_random_action()
        else:
            return self.__get_best_action(state)
        
    def __get_random_action(self):
        action_idx = random.randint(0, self.hparams.n_actions-1)
        action_vec = self.__get_action_vec(action_idx)
        return action_vec

    def __get_best_action(self, state):
        state = self.__wrap_state_into_tensors(state)
        print(state.shape)
        qvalues = self.model(state)[0]
        best_action_idx = torch.argmax(qvalues).item()
        best_action_vec = self.__get_action_vec(best_action_idx)
        return best_action_vec
    
    def __wrap_state_into_tensors(self, state):
        screen_buffer = state
        screen_buffer = torch.tensor([screen_buffer], device=self.device).float()
        return screen_buffer
    
    def __update_weights(self):
        self.model.load_state_dict(self.target_model.state_dict())
    
    def __play_step(self):
        action = self.get_action(self.state)
        self.env.make_action(action)
        reward = self.env.get_last_reward()
        next_state = self.env.get_state()
        done = self.env.is_episode_finished()
        self.buffer.add(self.state, action, reward, next_state, done)
        self.state = next_state
        if done:
            self.env.new_episode()
            self.state = self.env.get_state()
        return reward, done
    
    def __get_action_vec(self, action_idx):
        action_vector = [0] * self.hparams.n_actions
        action_vector[action_idx] = 1
        return action_vector
    
    def __populate_buffer(self):
        while len(self.buffer) < self.hparams.populate_steps:
            self.__play_step()
        self.env.new_episode()
        self.state = self.env.get_state()
        
    def on_fit_start(self):
        super().on_fit_start()
        self.env.init()
        self.state = self.env.get_state()
        self.__populate_buffer()
        
    def on_fit_end(self):
        super().on_fit_end()
        self.env.close()
        
    def training_step(self, batch, batch_no):
        for _ in range(10):
            reward, done = self.__play_step()
            self.episode_reward += reward
            
            if done:
                self.total_reward = self.episode_reward
                self.episode_reward = 0
        
        loss = self.__calculate_loss(batch)
        
        if self.global_step % self.hparams.update_weights_interval == 0:
            self.__update_weights()
            
        self.log('epsilon', self.hparams.epsilon)
        self.log('total_reward', self.total_reward)
        self.log('buffer_size', len(self.buffer))
        self.log('loss', loss)
        
    def configure_optimizers(self):
        optimizer = Adam(self.target_model.parameters(), lr=0.0001)
        return optimizer
    
    def train_dataloader(self):
        dataset = ReplayDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader
    
    def __calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = states.float()
        rewards = rewards.float()
        next_states = next_states.float()
        
        tmp = actions.long()
        state_action_values = self.target_model(states)
        print('+++', state_action_values.shape, tmp.shape)
        state_action_values = state_action_values.gather(1, tmp)
        state_action_values = state_action_values.squeeze(-1)

        with torch.no_grad():
            next_state_values = self.model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        print(state_action_values.shape, expected_state_action_values.shape)
        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    network = Network(10)
    summary(network, input_size=(32, 1, 320, 240))