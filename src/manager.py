from multiprocessing import Process, cpu_count
from random import choice
from time import sleep

from game import RewardedDoomGame
from config import GameConfig


actions = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]


class InstancesManager:
    
    def __init__(self, game_config, host_player, gui_players, bot_players):
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
                print('step', player.name)
                # game.advance_action()

                if game.is_player_dead():
                    game.respawn_player()

            game.new_episode()

        game.close()
        
    def __player_host(self):
        game = self.__create_game(self.host_player)
        game.add_game_args(f'-host {len(self.gui_players)+1}')
        game.init()
        return game
        
    def __create_game(self, player):
        game = RewardedDoomGame(player.rewards_config)
        self.game_config.setup_game(game)
        player.setup_game(game)
        return game
        
    def play(self):
        for client_player in self.gui_players:
            process = Process(target=self._player_client, args=(client_player,)).start()
            self.processes.append(process)
            
        host_game = self.__player_host()
        while True:
            host_game.make_action(choice(actions))
            print('step host')
            sleep(0.5)


if __name__ == '__main__':
    from players import AgentPlayer, BotPlayer
    
    config = GameConfig('cig.cfg')
    
    host = AgentPlayer('klima7')
    
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
    
    manager = InstancesManager(config, host, gui_players, bots)
    manager.play()
    sleep(5000)