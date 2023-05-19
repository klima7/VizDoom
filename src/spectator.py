import cv2
import numpy as np
import vizdoom as vzd

from lib.dqn import DQNPreprocessGameWrapper
from lib.setup import setup_multiplayer_game

game = setup_multiplayer_game()
game.set_mode(vzd.Mode.SPECTATOR)
game.init()

while True:
    while not game.is_episode_finished():
        state = game.get_state()
        
        cv2.imshow('Automap', np.transpose(state['automap'], [1, 2, 0]))
        cv2.imshow('Screen', state['screen'][0])
        cv2.imshow('Depth', state['screen'][1])
        for i, label_name in enumerate(DQNPreprocessGameWrapper.IMPORTANT_LABELS):
            cv2.imshow(f'Label {label_name}', state['screen'][2+i])
        cv2.waitKey(1)
        print(state['variables'])

        game.advance_action()
        
    game.new_episode()
