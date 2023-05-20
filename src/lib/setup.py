from pathlib import Path

import vizdoom as vzd

from lib.agent import RandomAgent
from lib.multi import Player, MultiplayerDoomWrapper, Bot
from lib.dqn import DQNPreprocessGameWrapper
from lib.reward import RewardsDoomWrapper, Rewards
    
    
def setup_multiplayer_game(log_rewards=False):
    player1_agent = RandomAgent(n_actions=10)
    player1_game = _create_game(name='player1', window_visible=False)
    player1 = Player(player1_agent, player1_game)
    
    player2_agent = RandomAgent(n_actions=10)
    player2_game = _create_game(name='player2')
    player2 = Player(player2_agent, player2_game)
    
    players = [player1, player2]
    bots = [Bot(), Bot(), Bot()]
    
    game = _apply_game_wrappers(_create_game(name='AI', window_visible=True), log_rewards=log_rewards)
    multiplayer_game = MultiplayerDoomWrapper(game, players, bots)
    return multiplayer_game


def _create_game(name, window_visible=False):
    game = vzd.DoomGame()
    
    game.load_config(str(Path(__file__).parent.parent.parent / 'scenarios' / 'cig.cfg'))
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args('-deathmatch')
    game.set_episode_timeout(5000)
    
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
        death_penalty=1,
        single_death_penalty=50,
        suicide_penalty=50,
        damage_reward=1,
        damage_penalty=1,
        health_reward=0,
        armor_reward=0,
        ammo_reward=0
    )
    game = RewardsDoomWrapper(game, rewards, log=log_rewards)
    game = DQNPreprocessGameWrapper(game)
    return game
