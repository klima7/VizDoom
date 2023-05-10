from random import choice

from rewards import RewardedDoomGame
from configs import GameConfig, RewardsConfig, AgentConfig, BotConfig
from multi import MultiDoomGame


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
    timeout=10
)

host = AgentConfig('klima7', rewards_config=rewards_config, window_visible=False)

gui_players = [
    AgentConfig('oponent1'),
    AgentConfig('oponent2'),
]

bots = [
    BotConfig(),
    BotConfig(),
    BotConfig(),
    BotConfig(),
]

game = MultiDoomGame(config, host, gui_players, bots, log_rewards=True)
game.init()

for i in range(1):
    while not game.is_episode_finished():
        game.make_action(choice(actions))
        print('step host')

    game.new_episode()
    print('new_episode host')
    
game.close()
print('finish')


game.init()

for i in range(1):
    while not game.is_episode_finished():
        game.make_action(choice(actions))
        print('step host')

    game.new_episode()
    print('new_episode host')
    
game.close()
print('finish')