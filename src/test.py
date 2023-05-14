from random import choice
from time import sleep

from lib.reward import MultiDoomGame
from lib.configs import GameConfig, RewardsConfig, Player, Bot
from lib.agent import RandomAgent


rewards_config = RewardsConfig(
    damage_reward=1,
    damage_penalty=1,
    death_penalty=50,
    single_death_penalty=200
)

game_config = GameConfig(
    config_name='cig.cfg',
    timeout=None
)

random_agent = RandomAgent()

host_config = Player(
    name='AI',
    agent=random_agent,
    rewards_config=rewards_config,
    window_visible=True
)

agent_configs = [
    Player(
        name='oponent1',
        agent=random_agent,
        window_visible=False
    ),
    Player(
        name='oponent2',
        agent=random_agent,
        window_visible=False
    ),
]

bots_configs = [
    Bot(),
]

game = MultiDoomGame(game_config, host_config, agent_configs, bots_configs)


def play(game):
    game.init()
    
    actions = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]

    for i in range(1):
        while not game.is_episode_finished():
            game.make_action(choice(actions))
            # sleep(1/60)
        game.new_episode()
        
    game.close()
    print('finish')


play(game)
