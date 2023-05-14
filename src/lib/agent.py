import itertools as it
from abc import ABC, abstractmethod
from random import choice

import numpy as np


class Agent(ABC):

    @abstractmethod    
    def get_action(self, game_state):
        pass
    
    def init(self, game):
        pass

    def reset(self):
        pass


class RandomAgent(Agent):
    
    def __init__(self, n_actions, action_duration=40):
        super().__init__()
        self.action_duration = action_duration
        self.actions = self.__create_actions(n_actions)
        
        self.action = None
        self.action_tic = 0
        
        
    def __create_actions(self, n_actions):
        actions = np.zeros((n_actions, n_actions))
        np.fill_diagonal(actions, 1)
        return actions
        
    
    def get_action(self, game_state):
        if self.action is None or game_state.tic > self.action_tic + self.action_duration:
            self.action = choice(self.actions)
            self.action_tic = game_state.tic
        return self.action
