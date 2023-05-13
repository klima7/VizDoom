import sys
from multiprocessing import Process

from .reward import RewardedDoomGame


class MultiDoomGame(RewardedDoomGame):
    
    def __init__(
        self,
        game_config,
        host_config,
        agent_configs,
        bot_configs,
        ):
        super().__init__(host_config.rewards_config, log=host_config.log_rewards)
        self.game_config = game_config
        self.host_config = host_config
        self.agent_configs = agent_configs
        self.bot_configs = bot_configs
        self.processes = []
        
        self.__config_host_game()
        
    def init(self):
        self.__start_agents_games()
        super().init()
        self.__add_bots()
        
    def close(self):
        super().close()
        self.__stop_agents_games()
        
    def __start_agents_games(self):
        for agent_config in self.agent_configs:
            process = Process(target=self.__agent_process, args=(agent_config,))
            process.start()
            self.processes.append(process)
            
    def __stop_agents_games(self):
        for process in self.processes:
            process.terminate()
            process.join()
        self.processes = []

    def __agent_process(self, agent_config):
        sys.stdout = None
        sys.stderr = None
        
        game = RewardedDoomGame(
            rewards_config = agent_config.rewards_config,
            log = agent_config.log_rewards
        )
        self.__config_client_game(game, agent_config)
        game.init()
        agent_config.agent.init(game)
        
        while True:
            while not game.is_episode_finished():
                state = game.get_state()
                action = agent_config.agent.get_action(state)
                game.make_action(action)
            game.new_episode()
            
    def __config_host_game(self):
        self.game_config.setup_game(self)
        self.host_config.setup_game(self)
        self.add_game_args(f'-host {len(self.agent_configs)+1}')

    def __config_client_game(self, game, agent_config):
        self.game_config.setup_game(game)
        agent_config.setup_game(game)
        game.add_game_args('-join 127.0.0.1')

    def __add_bots(self):
        for bot_config in self.bot_configs:
            if bot_config.name:
                self.send_game_command(f'addbot {bot_config.name}')
            else:
                self.send_game_command('addbot')
