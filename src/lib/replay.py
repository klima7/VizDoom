import random

import numpy as np
from torch.utils.data.dataset import IterableDataset


class ReplayBuffer(object):
    
    def __init__(self, size):
        self.__storage = []
        self.__maxsize = size
        self.__next_idx = 0

    def __len__(self):
        return len(self.__storage)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, np.float32(reward), next_state, done)

        if self.__next_idx >= len(self.__storage):
            self.__storage.append(data)
        else:
            self.__storage[self.__next_idx] = data
        self.__next_idx = (self.__next_idx + 1) % self.__maxsize

    def sample(self):
        idx = random.randint(0, len(self.__storage) - 1)
        return self.__storage[idx]


class ReplayDataset(IterableDataset):
    
    def __init__(self, replay, game):
        self.replay = replay
        self.game = game

    def __iter__(self):
        self.game.new_episode()
        while not self.game.is_episode_finished():
            state, action, reward, next_state, done = self.replay.sample()

            state = {
                'screen': state['screen'].astype(np.float32) / 255,
                'variables': state['variables']
            }

            next_state = {
                'screen': next_state['screen'].astype(np.float32) / 255,
                'variables': next_state['variables']
            }

            yield state, action, reward, next_state, done
