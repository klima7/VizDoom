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
        self.players = players
        self.bots = bots
        
        self.players_processes = []
        self.game = game
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
        for player in self.players:
            process = Process(target=self.__agent_process, args=(player,))
            process.start()
            self.players_processes.append(process)
            
    def __stop_agents_games(self):
        for process in self.players_processes:
            process.terminate()
            process.join()
        self.players_processes = []

    def __agent_process(self, player):
        agent = player.agent
        game = player.game
        
        game.add_game_args('-join 127.0.0.1')
        game.init()
        
        while True:
            while not game.is_episode_finished():
                state = game.get_state()
                action = agent.get_action(state)
                game.make_action(action)
            game.new_episode()

    def __add_bots(self):
        for bot in self.bots:
            if bot.name:
                self.send_game_command(f'addbot {bot.name}')
            else:
                self.send_game_command('addbot')
