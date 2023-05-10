from abc import ABC, abstractmethod
from random import choice


class Agent(ABC):

    @abstractmethod    
    def get_action(self, game_state):
        pass

    def reset(self):
        pass


class RandomAgent(Agent):
    
    def get_action(self, game_state):
        actions = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ]
        return choice(actions)
    