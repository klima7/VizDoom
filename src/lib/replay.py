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
        data = (state, np.array(action), np.float32(reward), next_state, np.array(done))

        if self.__next_idx >= len(self.__storage):
            self.__storage.append(data)
        else:
            self.__storage[self.__next_idx] = data
        self.__next_idx = (self.__next_idx + 1) % self.__maxsize

    def sample(self):
        idx = random.randint(0, len(self.__storage) - 1)
        return self.__storage[idx]


class ReplayDataset(IterableDataset):
    
    def __init__(self, replay, batch_size):
        self.replay = replay
        self.batch_size = batch_size
        self.__end_epoch = False
        
    def __iter__(self):
        while not self.__end_epoch:
            sample = self.replay.sample()
            sample[0]['screen'] = sample[0]['screen'].astype(np.float32) / 255
            sample[3]['screen'] = sample[3]['screen'].astype(np.float32) / 255
            yield sample
        self.__end_epoch = False

    def end_epoch(self):
        self.__end_epoch = True
