from random import choice

from games import MultiDoomGame
from configs import GameConfig, RewardsConfig, AgentConfig, BotConfig
from agent import RandomAgent


actions = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
]

rewards_config = RewardsConfig(
    damage_reward=1,
    damage_penalty=1,
    death_penalty=50,
    single_death_penalty=200
)

config = GameConfig(
    config_name='cig.cfg',
    timeout=1000
)

random_agent = RandomAgent()

host = AgentConfig(
    name='klima7',
    agent=random_agent,
    rewards_config=rewards_config,
    window_visible=True
)

gui_players = [
    AgentConfig(
        name='oponent1',
        agent=random_agent,
    ),
    AgentConfig(
        name='oponent2',
        agent=random_agent,
    ),
]

bots = [
    BotConfig(),
    BotConfig(),
    BotConfig(),
    BotConfig(),
]

game = MultiDoomGame(config, host, gui_players, bots, log_rewards=False)
game.init()

for i in range(1):
    while not game.is_episode_finished():
        game.make_action(choice(actions))
    game.new_episode()
    
game.close()
print('finish')
