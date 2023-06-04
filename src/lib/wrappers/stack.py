import numpy as np
import cv2


class StackStateGameWrapper:

    def __init__(self, game, n_frames=5):
        self.game = game
        self.__n_frames = n_frames

        self.__screens = None
        self.__vars = None

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__clear_frames()

    def new_episode(self):
        self.game.new_episode()
        self.__clear_frames()

    def get_screen_size(self):
        screen_size = list(self.game.get_screen_size())
        screen_size[0] *= self.__n_frames
        return tuple(screen_size)

    def get_variables_size(self):
        variables_size = self.game.get_variables_size()
        variables_size *= self.__n_frames
        return variables_size

    def get_state(self):
        state = self.game.get_state()

        # update screens
        self.__screens = self.__screens[1:, ...]
        self.__screens = np.concatenate([self.__screens, state['screen']], axis=0)

        # update variables
        n_vars = len(state['variables'])
        self.__vars = self.__vars[n_vars:]
        self.__vars = np.concatenate([self.__vars, state['variables']], axis=0)

        assert self.__screens.shape == self.get_screen_size()
        assert self.__vars.shape == (self.get_variables_size(),)

        return {
            'screen': self.__screens,
            'variables': self.__vars
        }

    def __clear_frames(self):
        self.__screens = np.zeros(self.get_screen_size(), dtype=np.uint8)
        self.__vars = np.zeros(self.get_variables_size(), dtype=np.uint8)
