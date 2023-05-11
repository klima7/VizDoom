import os
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import vizdoom as vzd

from agent import Agent


@dataclass(kw_only=True)
class GameConfig:
    config_name: str
    map: str | None = None
    timeout: int | None = None
    respawn_delay: int = 0
    ticrate: int | None = None
    mode: vzd.Mode = vzd.Mode.PLAYER

    def setup_game(self, game):
        game.load_config(str(Path(__file__).parent.parent / 'scenarios' / self.config_name))
        game.set_mode(self.mode)
        game.add_game_args('-deathmatch')
        
        if self.map:
            game.set_doom_map(self.map)
        if self.timeout:
            game.set_episode_timeout(self.timeout)
        if self.ticrate:
            game.set_ticrate(self.ticrate)


@dataclass(kw_only=True)
class RewardsConfig:
    live_reward: float = 0
    kill_reward: float = 0
    hit_reward: float = 0
    damage_reward: float = 0
    health_reward: float = 0
    armor_reward: float = 0
    item_reward: float = 0
    secret_reward: float = 0
    attack_reward: float = 0
    alt_attack_reward: float = 0
    death_penalty: float = 0
    single_death_penalty: float = 0
    suicide_penalty: float = 0
    hit_penalty: float = 0
    damage_penalty: float = 0
    health_penalty: float = 0
    armor_penalty: float = 0
    attack_penalty: float = 0
    alt_attack_penalty: float = 0


@dataclass
class BotConfig:
    name: str | None = None


@dataclass
class AgentConfig:
    name: str
    agent: Agent | None = None
    rewards_config: RewardsConfig = RewardsConfig()
    log_rewards: bool = False
    window_visible: bool = False
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_320X240
    screen_format: vzd.ScreenFormat = vzd.ScreenFormat.GRAY8
    console_enabled: bool = False
        
    def setup_game(self, game):
        game.add_game_args(f'+name {self.name}')
        game.set_screen_resolution(self.screen_resolution)
        game.set_window_visible(self.window_visible)
        game.set_console_enabled(self.console_enabled)
        game.set_screen_format(self.screen_format)
