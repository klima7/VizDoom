from multiprocessing import Process
from random import choice
from rewards import RewardedDoomGame


class MultiDoomGame(RewardedDoomGame):
    
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
        
    def init(self):
        self.__start_associated_games()
        super().init()
        
    def close(self):
        super().close()
        self.__stop_associated_games()
        
    def __start_associated_games(self):
        for client_player in self.gui_players:
            process = Process(target=self.__player_client, args=(client_player,))
            process.start()
            self.processes.append(process)
            
    def __stop_associated_games(self):
        for process in self.processes:
            process.kill()
            process.join()
        self.processes = []

    def __player_client(self, player):
        game = RewardedDoomGame(player.rewards_config)
        self.game_config.setup_game(game)
        player.setup_game(game)
        game.add_game_args('-join 127.0.0.1')
        game.init()
        
        while True:
            while not game.is_episode_finished():
                state = game.get_state()
                action = player.agent.get_action(state)
                game.make_action(action)
                if game.is_player_dead():
                    game.respawn_player()
            game.new_episode()
