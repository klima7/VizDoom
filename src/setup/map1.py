from pathlib import Path

import vizdoom as vzd

from lib.dqn import DQNPreprocessGameWrapper
from lib.wrappers.reward import RewardsDoomWrapper, Rewards
from lib.wrappers.bots import AddBotsDoomWrapper
from lib.wrappers.map import SingleMapDoomWrapper
    
    
def setup_game(name='AI', log_rewards=False, window_visible=True):
    game = _create_base_game(name, window_visible)
    game = _apply_game_wrappers(game, log_rewards=log_rewards)
    return game


def _create_base_game(name, window_visible=False):
    game = vzd.DoomGame()
    
    game.load_config(str(Path(__file__).parent.parent.parent / 'scenarios' / 'multi.cfg'))
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args('-deathmatch')
    game.set_episode_timeout(3000)
    
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    
    # game.set_automap_buffer_enabled(True)
    # game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    # game.set_automap_rotate(True)
    # game.set_automap_render_textures(False)

    game.set_render_hud(False)
    game.set_render_minimal_hud(True)
    game.set_render_particles(False)
    game.set_render_decals(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(False)
    game.set_render_weapon(False)
    
    game.add_game_args(f'+name {name}')
    game.set_window_visible(window_visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_console_enabled(False)
    return game


def _apply_game_wrappers(game, log_rewards):
    rewards = Rewards(
        kill_reward=20,
        hit_reward=5,
        hit_penalty=1,
        damage_reward=1,
        damage_penalty=0.2,
    )
    game = RewardsDoomWrapper(game, rewards, log=log_rewards)
    game = SingleMapDoomWrapper(game, map='map01')
    game = AddBotsDoomWrapper(game, bots_count=4)
    game = DQNPreprocessGameWrapper(game)
    return game
