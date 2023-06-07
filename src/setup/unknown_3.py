from pathlib import Path

import vizdoom as vzd

from lib.wrappers import AddBotsDoomWrapper, SetMonstersDoomWrapper, ModifyButtonsWrapper, \
    RewardsDoomWrapper, Rewards, PreprocessGameWrapper, StackStateGameWrapper


def setup_game(name='AI', log_rewards=False, window_visible=True):
    game = _create_base_game(name, window_visible)
    game = _apply_game_wrappers(game, log_rewards=log_rewards)
    return game


def _create_base_game(name, window_visible=False):
    game = vzd.DoomGame()

    game.load_config(str(Path(__file__).parent.parent.parent / 'scenarios' / 'cig_with_unknown' / 'cig_with_unknown.cfg'))
    game.set_doom_map("map03")
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args('-deathmatch ')
    game.add_game_args('+viz_bots_path ../bots/stupid.cfg ')
    game.add_game_args('+sv_forcerespawn 1 ')
    game.add_game_args('+sv_noautoaim 1 ')
    game.add_game_args('+sv_respawnprotect 1 ')
    game.add_game_args('+sv_spawnfarthest 1 ')
    game.add_game_args('+sv_nocrouch 1 ')
    game.add_game_args('+viz_nocheat 1 ')
    game.add_game_args('+sv_respawnprotect 1 ')
    game.add_game_args('+viz_respawn_delay 10 ')
    game.add_game_args('+sv_noexit 1 ')
    game.add_game_args('+sv_samelevel 1 ')
    game.add_game_args('+alwaysapplydmflags 1 ')

    game.set_episode_timeout(10_000)
    game.set_doom_skill(1)

    game.set_render_hud(True)
    game.set_render_minimal_hud(False)
    game.set_render_particles(False)
    game.set_render_decals(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(False)
    game.set_render_weapon(True)

    game.add_game_args(f'+name {name}')
    game.set_window_visible(window_visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_console_enabled(False)
    return game


def _apply_game_wrappers(game, log_rewards):
    rewards = Rewards(
        kill_reward=40,
        # single_death_penalty=20,
        hit_reward=1,
        # hit_penalty=1,
        # damage_reward=0,
        # damage_penalty=0.3,
        # ammo_penalty=0.5,
        # health_reward=0.3,
        # armor_reward=0.3,
        # ammo_reward=0.3,
        # ammo_penalty=0.2,
        suicide_penalty=-10
    )

    labels = []

    variables = {
        vzd.GameVariable.HEALTH: slice(0, 100),
        vzd.GameVariable.ARMOR: slice(0, 200),
    }

    game = RewardsDoomWrapper(game, rewards, log=log_rewards)
    game = AddBotsDoomWrapper(game, bots_count=15, difficulty=1)
    game = SetMonstersDoomWrapper(game, monsters_count=20)
    game = PreprocessGameWrapper(
        game,
        screen_size=(40, 60),
        labels=labels,
        variables=variables,
        depth=False
    )
    game = ModifyButtonsWrapper(
        game,
        digital_buttons=[
            None,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
        ],
        delta_buttons={
            # vzd.Button.LOOK_UP_DOWN_DELTA: (-1, 1)
        }
    )
    game = StackStateGameWrapper(game, n_frames=5)
    return game
