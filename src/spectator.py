import cv2
import vizdoom as vzd

from setup import setup_game

game = setup_game(log_rewards=True)
game.set_mode(vzd.Mode.SPECTATOR)
game.init()

while True:
    while not game.is_episode_finished():
        state = game.get_state()

        print(state['variables'])
        screen = state['screen']
        for i, channel in enumerate(screen):
            cv2.imshow(f'screen {i}', channel)
        cv2.waitKey(1)

        game.advance_action()

    game.new_episode()
    print('episode')
