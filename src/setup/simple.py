from pathlib import Path

import vizdoom as vzd

from lib.dqn import DQNPreprocessGameWrapper
from lib.wrappers.reward import RewardsDoomWrapper, Rewards


def setup_game(name='AI', log_rewards=False, window_visible=True):
    game = _create_base_game(name, window_visible)
    game = _apply_game_wrappers(game, log_rewards=log_rewards)
    return game


def _create_base_game(name, window_visible=False):
    game = vzd.DoomGame()

    game.load_config(str(Path(__file__).parent.parent.parent / 'scenarios' / 'simpler_basic.cfg'))
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    return game


def _apply_game_wrappers(game, log_rewards):
    rewards = Rewards(
        life_reward=-1,
        hit_reward=100,
    )
    game = RewardsDoomWrapper(game, rewards, log=log_rewards)
    game = DQNPreprocessGameWrapper(game)
    return game
