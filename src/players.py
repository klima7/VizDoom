from abc import ABC

import vizdoom as vzd

from game import RewardsConfig


class PlayerConfig(ABC):
    
    def __init__(self, name):
        self.name = name


class BotConfig(PlayerConfig):
    
    def __init__(self, name=None):
        super().__init__(name)


class AgentConfig(PlayerConfig):
    
    def __init__(
        self,
        name,
        rewards_config = RewardsConfig(),
        window_visible = False,
        screen_resolution = vzd.ScreenResolution.RES_320X240,
        console_enabled = False
        ):
        super().__init__(name)
        self.rewards_config = rewards_config
        self.window_visible = window_visible
        self.screen_resolution = screen_resolution
        self.console_enabled = console_enabled
        
    def setup_game(self, game):
        game.add_game_args(f'+name {self.name}')
        game.set_screen_resolution(self.screen_resolution)
        game.set_window_visible(self.window_visible)
        game.set_console_enabled(self.console_enabled)
