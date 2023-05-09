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


class MultiplayerRewardedGame(RewardedDoomGame):
    
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
        
    def _player_client(self, player):
        game = self.__create_game(player)
        game.add_game_args('-join 127.0.0.1')
        
        game.init()
        
        for i in range(10):
            while not game.is_episode_finished():
                state = game.get_state()
                game.make_action(choice(actions))
                # print('step', player.name)
                # game.advance_action()

                if game.is_player_dead():
                    game.respawn_player()

            game.new_episode()
            print('new_episode', player.name)

        game.close()
        
    def __create_game(self, player):
        game = RewardedDoomGame(player.rewards_config)
        self.game_config.setup_game(game)
        player.setup_game(game)
        return game
        
    def spin(self):
        for client_player in self.gui_players:
            process = Process(target=self._player_client, args=(client_player,)).start()
            self.processes.append(process)
            
        self.clear_game_args()
        self.game_config.setup_game(self)
        self.host_player.setup_game(self)
        self.add_game_args(f'-host {len(self.gui_players)+1}')
        self.init()


if __name__ == '__main__':
    from players import AgentPlayer, BotPlayer
    
    rewards_config = RewardsConfig(
        damage_reward=1,
        damage_penalty=1,
        death_penalty=50,
        single_death_penalty=200
    )
    
    config = GameConfig('cig.cfg', timeout=None)
    
    host = AgentPlayer('klima7', rewards_config=rewards_config, window_visible=True)
    
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
    
    game = MultiplayerRewardedGame(config, host, gui_players, bots, log_rewards=True)
    game.spin()
    
    for i in range(10):
        while not game.is_episode_finished():
            game.make_action(choice(actions))
            # print('step host')

        game.new_episode()
        print('new_episode host')
        
    print('finish')