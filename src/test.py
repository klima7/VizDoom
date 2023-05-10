from random import choice
from time import sleep

from games import MultiDoomGame
from configs import GameConfig, RewardsConfig, AgentConfig, BotConfig
from agent import RandomAgent


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

host_config = AgentConfig(
    name='klima7',
    agent=random_agent,
    rewards_config=rewards_config,
    window_visible=False
)

agent_configs = [
    AgentConfig(
        name='oponent1',
        agent=random_agent,
        window_visible=True
    ),
    AgentConfig(
        name='oponent2',
        agent=random_agent,
    ),
]

bots_configs = [
    BotConfig(),
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
            sleep(1/60)
        game.new_episode()
        
    game.close()
    print('finish')


play(game)
