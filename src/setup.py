from pathlib import Path

import vizdoom as vzd

from lib.dqn import DQNPreprocessGameWrapper
from lib.reward import RewardsDoomWrapper, Rewards
from lib.bots import AddBotsDoomWrapper
    
    
def setup_multiplayer_game(log_rewards=False):
    game = _apply_game_wrappers(_create_game(name='AI', window_visible=True), log_rewards=log_rewards)
    return game


def _create_game(name, window_visible=False):
    game = vzd.DoomGame()
    
    game.load_config(str(Path(__file__).parent.parent / 'scenarios' / 'multi.cfg'))
    game.set_doom_map('map01')
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args('-deathmatch')
    game.set_episode_timeout(60000)
    
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(True)
    game.set_automap_render_textures(False)

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
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_console_enabled(False)
    return game


def _apply_game_wrappers(game, log_rewards):
    rewards = Rewards(
        kill_reward=50,
        single_death_penalty=50,
        damage_reward=1,
        damage_penalty=1,
    )
    game = RewardsDoomWrapper(game, rewards, log=log_rewards)
    game = DQNPreprocessGameWrapper(game)
    game = AddBotsDoomWrapper(game, bots_count=1)
    return game
