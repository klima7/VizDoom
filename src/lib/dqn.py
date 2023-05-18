import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchinfo import summary

from .agent import Agent
from .replay import ReplayBuffer, ReplayDataset


class DQNPreprocessGameWrapper:
    
    IMPORTANT_LABELS = [
        232,    # RocketAmmo
        199,    # Medikit
        188,    # DoomPlayer
        236,    # ExplosiveBarrel
        246,    # Rocket
        170,    # BlueArmor
        162,    # GreenArmor
    ]
    
    def __init__(self, game, collect_labels=False):
        self.game = game
        self.__collect_labels = collect_labels
        self.__seen_labels = {}
        
    def __getattr__(self, attr):
        return getattr(self.game, attr)
    
    def get_state(self):
        state = self.game.get_state()
        
        if self.__collect_labels:
            self.__update_seen_labels(state)
        
        if self.game.is_episode_finished():
            w = self.game.get_screen_width()
            h = self.game.get_screen_height()
            c = self.game.get_screen_channels()
            
            screen_buffer = np.zeros((c, h, w)).astype(np.float32)
            depth_buffer = np.zeros((1, h, w)).astype(np.float32)
            automap_buffer = np.zeros((1, h, w)).astype(np.float32)
            labels = np.zeros((len(self.IMPORTANT_LABELS), h, w)).astype(np.float32)
        
        else:
            screen_buffer = state.screen_buffer[np.newaxis, ...]  # 240x320
            depth_buffer = state.depth_buffer[np.newaxis, ...]
            automap_buffer = state.automap_buffer[np.newaxis, ...]
            labels = self.__get_important_labels_map(state.labels_buffer)

        screen = np.concatenate([screen_buffer, depth_buffer, labels], axis=0)
        automap = automap_buffer
        
        return {
            'screen': screen.astype(np.float32),
            'automap': automap.astype(np.float32)
        }
    
    def __get_important_labels_map(self, labels_buffer):
        maps = []
        for important_label in self.IMPORTANT_LABELS:
            map = labels_buffer == important_label
            maps.append(map)
        return np.array(maps)
    
    def __update_seen_labels(self, state):
        if state:
            for label in state.labels:
                if label.object_name not in self.__seen_labels:
                    self.__seen_labels[label.object_name] = label.value
        print(f'labels ({len(self.__seen_labels)})', self.__seen_labels)


class DQNNetwork(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        
        self.screen_net = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.automap_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.neck_net = nn.Sequential(
            nn.Linear(1728*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, data):
        screen_out = self.screen_net(data['screen'])
        automap_out = self.automap_net(data['automap'])
        conv_out = torch.cat([screen_out, automap_out], axis=1)
        neck_out = self.neck_net(conv_out)
        return neck_out
    
    def forward_state(self, state, device=None):
        screens = torch.tensor(state['screen'], device=device, dtype=torch.float32).unsqueeze(0)
        automaps = torch.tensor(state['automap'], device=device, dtype=torch.float32).unsqueeze(0)
        data = {'screen': screens, 'automap': automaps}
        return self.forward(data)[0]


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
        self.model = DQNNetwork(self.hparams.n_actions)
        self.target_model = DQNNetwork(self.hparams.n_actions)
        
        self.buffer = ReplayBuffer(self.hparams.buffer_size)
        self.dataset = ReplayDataset(self.buffer, self.hparams.batch_size)
        self.state = None
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
        action_idx = random.randint(0, self.hparams.n_actions-1)
        action_vec = self.__get_action_vec(action_idx)
        return action_vec

    def __get_best_action(self, state):
        qvalues = self.model.forward_state(state, self.device)
        best_action_idx = torch.argmax(qvalues).item()
        best_action_vec = self.__get_action_vec(best_action_idx)
        return best_action_vec
    
    def __update_weights(self):
        self.model.load_state_dict(self.target_model.state_dict())
    
    def __play_step(self, reset=True):
        action = self.get_action(self.state)
        self.env.make_action(action)
        reward = self.env.get_last_reward()
        next_state = self.env.get_state()
        done = self.env.is_episode_finished()
        self.buffer.add(self.state, action, reward, next_state, done)
        self.state = next_state
        if done and reset:
            self.env.new_episode()
            self.state = self.env.get_state()
        return done
    
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
            done = self.__play_step(reset=False)
            
            if done:
                self.dataset.end_epoch()
                self.reset()
                self.__update_metrics()
                self.env.new_episode()
                self.state = self.env.get_state()
                
        loss = self.__calculate_loss(batch)
        
        if self.global_step % self.hparams.update_weights_interval == 0:
            self.__update_weights()
            
        self.__log_metrics(loss)
        return loss
    
    def __log_metrics(self, loss, prefix=''):
        self.log('loss', loss)
        self.log(prefix+'epsilon', float(self.hparams.epsilon))
        self.log(prefix+'buffer_size', float(len(self.buffer)))
        
        self.log(prefix+'total_reward', float(self.total_reward))
        self.log(prefix+'frags_count', float(self.frags_count))
        self.log(prefix+'suicides_count', float(self.suicides_count))
        self.log(prefix+'deaths_count', float(self.deaths_count))
        self.log(prefix+'hits_made_count', float(self.hits_made_count))
        self.log(prefix+'hits_taken_count', float(self.hits_taken_count))
        self.log(prefix+'items_collected_count', float(self.items_collected_count))
        self.log(prefix+'damage_make_count', float(self.damage_make_count))
        self.log(prefix+'damage_taken_count', float(self.damage_taken_count))
        self.log(prefix+'armor_gained_count', float(self.armor_gained_count))
        self.log(prefix+'armor_lost_count', float(self.armor_lost_count))
        self.log(prefix+'health_gained_count', float(self.health_gained_count))
        self.log(prefix+'health_lost_count', float(self.health_lost_count))
        self.log(prefix+'death_tics_count', float(self.death_tics_count))
        self.log(prefix+'attack_not_ready_tics', float(self.attack_not_ready_tics))
        
    def __update_metrics(self):
        self.total_reward = self.env.get_total_reward()
        self.frags_count = self.env.get_frags_count()
        self.suicides_count = self.env.get_suicides_count()
        self.deaths_count = self.env.get_deaths_count()
        self.hits_made_count = self.env.get_hits_made_count()
        self.hits_taken_count = self.env.get_hits_taken_count()
        self.items_collected_count = self.env.get_items_collected_count()
        self.damage_make_count = self.env.get_damage_make_count()
        self.damage_taken_count = self.env.get_damage_taken_count()
        self.armor_gained_count = self.env.get_secrets_count()
        self.armor_lost_count = self.env.get_armor_gained_count()
        self.health_gained_count = self.env.get_armor_lost_count()
        self.health_lost_count = self.env.get_health_gained_count()
        self.death_tics_count = self.env.get_health_lost_count()
        self.attack_not_ready_tics = self.env.get_death_tics_count()
        
    def configure_optimizers(self):
        optimizer = Adam(self.target_model.parameters(), lr=0.0001)
        return optimizer
    
    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size
        )
        return dataloader
    
    def __calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = torch.argmax(actions, axis=1)
        
        state_action_values = self.target_model(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    network = DQNNetwork(10)
    summary(network, input_size=(32, 1, 320, 240))
