from multiprocessing import Process, cpu_count
from random import choice
from time import sleep

from game import RewardedDoomGame, RewardsConfig
from config import GameConfig


actions = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]


class MultiplayerDoomGame(RewardedDoomGame):
    
    def __init__(
        self,
        game_config,
        host_player,
        gui_players,
        bot_players,
        log_rewards=False
        ):
        super().__init__(host_player.rewards_config, log_rewards)
        self.game_config = game_config
        self.host_player = host_player
        self.gui_players = gui_players
        self.bot_players = bot_players
        self.processes = []
        
        self.game_config.setup_game(self)
        self.host_player.setup_game(self)
        self.add_game_args(f'-host {len(self.gui_players)+1}')
        
    def _player_client(self, player):
        game = RewardedDoomGame(player.rewards_config)
        self.game_config.setup_game(game)
        player.setup_game(game)
        game.add_game_args('-join 127.0.0.1')
        
        game.init()
        
        for i in range(10):
            while not game.is_episode_finished():
                state = game.get_state()
                game.make_action(choice(actions))
                print('step', player.name)

                if game.is_player_dead():
                    game.respawn_player()

            game.new_episode()
            print('new_episode', player.name)

        game.close()
        
    def init(self):
        for client_player in self.gui_players:
            process = Process(target=self._player_client, args=(client_player,))
            process.start()
            self.processes.append(process)
        super().init()
        
    def close(self):
        super().close()
        for process in self.processes:
            process.kill()
            process.join()
        self.processes = []


if __name__ == '__main__':
    from players import AgentPlayer, BotPlayer
    
    rewards_config = RewardsConfig(
        damage_reward=1,
        damage_penalty=1,
        death_penalty=50,
        single_death_penalty=200
    )
    
    config = GameConfig('cig.cfg', timeout=10)
    
    host = AgentPlayer('klima7', rewards_config=rewards_config, window_visible=False)
    
    gui_players = [
        AgentPlayer('oponent1'),
        AgentPlayer('oponent2'),
    ]
    
    bots = [
        BotPlayer(),
        BotPlayer(),
        BotPlayer(),
        BotPlayer(),
    ]
    
    game = MultiplayerDoomGame(config, host, gui_players, bots, log_rewards=True)
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