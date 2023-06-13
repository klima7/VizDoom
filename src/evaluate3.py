from time import sleep

from lib.dqn import DQNAgent
from setup.unknown_3 import *

agent = DQNAgent.load_from_checkpoint('../best_checkpoints/dqn3.ckpt')

game = setup_game('AI')
game.init()

for _ in range(20):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        best_action_index = agent.get_action(state, epsilon=0.1)

        game.make_action(best_action_index, skip=3)

    sleep(0)
    frags_count = game.get_frags_count()
    damage = game.get_damage_make_count()
    deaths = game.get_deaths_count()
    print(f"Frags: {frags_count}; Damage: {damage}; Deaths: {deaths}")
