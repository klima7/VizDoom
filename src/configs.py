import os
from abc import ABC

import vizdoom as vzd


class GameConfig:
    
    def __init__(
        self,
        config_name,
        map = None,
        timeout = None,
        respawn_delay = 0,
        ticrate = None,
        mode = vzd.Mode.PLAYER
        ):
        self.config_name = config_name
        self.map = map
        self.timeout = timeout
        self.respawn_delay = respawn_delay
        self.ticrate = ticrate
        self.mode = mode

    def setup_game(self, game):
        game.load_config(os.path.join(vzd.scenarios_path, self.config_name))
        game.set_mode(self.mode)
        game.add_game_args('-deathmatch')
        
        if self.map:
            game.set_doom_map(self.map)
        if self.timeout:
            game.set_episode_timeout(self.timeout)
        if self.ticrate:
            game.set_ticrate(self.ticrate)


class RewardsConfig:
    
    def __init__(
        self,
        live_reward = 0,
        kill_reward = 0,
        hit_reward = 0, 
        damage_reward = 0,
        health_reward = 0,
        armor_reward = 0,
        item_reward = 0,
        secret_reward = 0,
        attack_reward = 0,
        alt_attack_reward = 0,
        death_penalty = 0,
        single_death_penalty = 0,
        suicide_penalty = 0,
        hit_penalty = 0,
        damage_penalty = 0,
        health_penalty = 0,
        armor_penalty = 0,
        attack_penalty = 0,
        alt_attack_penalty = 0,
    ):
        self.live_reward = live_reward
        self.kill_reward = kill_reward
        self.hit_reward = hit_reward 
        self.damage_reward = damage_reward
        self.health_reward = health_reward
        self.armor_reward = armor_reward
        self.item_reward = item_reward
        self.secret_reward = secret_reward
        self.attack_reward = attack_reward
        self.alt_attack_reward = alt_attack_reward
        self.death_penalty = death_penalty
        self.single_death_penalty = single_death_penalty
        self.suicide_penalty = suicide_penalty
        self.hit_penalty = hit_penalty
        self.damage_penalty = damage_penalty
        self.health_penalty = health_penalty
        self.armor_penalty = armor_penalty
        self.attack_penalty = attack_penalty
        self.alt_attack_penalty = alt_attack_penalty


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
        agent = None,
        rewards_config = RewardsConfig(),
        window_visible = False,
        screen_resolution = vzd.ScreenResolution.RES_320X240,
        console_enabled = False
        ):
        super().__init__(name)
        self.agent = agent
        self.rewards_config = rewards_config
        self.window_visible = window_visible
        self.screen_resolution = screen_resolution
        self.console_enabled = console_enabled
        
    def setup_game(self, game):
        game.add_game_args(f'+name {self.name}')
        game.set_screen_resolution(self.screen_resolution)
        game.set_window_visible(self.window_visible)
        game.set_console_enabled(self.console_enabled)
