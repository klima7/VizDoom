import sys
from dataclasses import dataclass
from multiprocessing import Process

import vizdoom as vzd

from .agent import Agent


@dataclass
class Bot:
    name: str | None = None


@dataclass
class Player:
    agent: Agent | None = None
    game: vzd.DoomGame = None


class MultiplayerDoomWrapper:
    
    def __init__(self, game, players, bots):
        self.game = game
        self.__players = players
        self.__bots = bots
        self.__players_processes = []
        self.game.add_game_args(f'-host {len(players)+1}')
        
    def __getattr__(self, attr):
        return getattr(self.game, attr)
        
    def init(self):
        self.__start_agents_games()
        self.game.init()
        self.__add_bots()
        
    def close(self):
        self.game.close()
        self.__stop_agents_games()
        
    def __start_agents_games(self):
        for player in self.__players:
            process = Process(target=self.__agent_process, args=(player,))
            process.start()
            self.__players_processes.append(process)
            
    def __stop_agents_games(self):
        for process in self.__players_processes:
            process.terminate()
            process.join()
        self.__players_processes = []

    def __agent_process(self, player):
        agent = player.agent
        game = player.game
        
        game.add_game_args('-join 127.0.0.1')
        game.init()
        game.set_doom_map('map01')
        
        while True:
            while not game.is_episode_finished():
                state = game.get_state()
                action = agent.get_action(state)
                game.make_action(action)
            agent.reset()
            game.new_episode()
            game.set_doom_map('map01')

    def __add_bots(self):
        for bot in self.__bots:
            if bot.name:
                self.send_game_command(f'addbot {bot.name}')
            else:
                self.send_game_command('addbot')
