from lib.dqn import DQNAgent
from setup.unknown_1 import *


agent = DQNAgent.load_from_checkpoint('../best_checkpoints/dqn.ckpt')

game = setup_game(name="Łukasz", window_visible=True)
game.add_game_args("-join 127.0.0.1 -port 5029")
game.init()

while not game.is_episode_finished():
    state = game.get_state()
    action = agent.get_action(state, epsilon=0)
    game.make_action(action, skip=3)

    if game.is_player_dead():
        game.respawn_player()

game.close()
