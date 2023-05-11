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
        data = (state, action, reward, next_state, done)

        if self.__next_idx >= len(self.__storage):
            self.__storage.append(data)
        else:
            self.__storage[self.__next_idx] = data
        self.__next_idx = (self.__next_idx + 1) % self.__maxsize

    def __encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self.__storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones)
        )

    def sample(self, batch_size):
        idxes = [
            random.randint(0, len(self.__storage) - 1)
            for _ in range(batch_size)
        ]
        return self.__encode_sample(idxes)


class ReplayDataset(IterableDataset):
    
    def __init__(self, replay, batch_size):
        self.replay = replay
        self.batch_size = batch_size
        
    def __iter__(self):
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        for data in zip(states, actions, rewards, next_states, dones):
            yield data
