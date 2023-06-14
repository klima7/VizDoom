from pathlib import Path

import vizdoom as vzd

from lib.wrappers import AddBotsDoomWrapper, ModifyButtonsWrapper, \
    RewardsDoomWrapper, Rewards, PreprocessGameWrapper, StackStateGameWrapper

from lib.dqn import DQNAgent


def setup_game(name='AI', log_rewards=False, window_visible=True):
    game = _create_base_game(name, window_visible)
    game = _apply_game_wrappers(game, log_rewards=log_rewards)
    return game


def _create_base_game(name, window_visible=False):
    game = vzd.DoomGame()

    game.load_config(str(Path(__file__).parent.parent / 'scenarios' / 'cig_with_unknown' / 'cig_with_unknown.cfg'))
    game.set_doom_map("map01")
    game.set_mode(vzd.Mode.PLAYER)

    # rendering
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_particles(False)
    game.set_render_decals(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(False)
    game.set_render_weapon(True)
    game.set_render_crosshair(False)
    game.set_render_effects_sprites(False)

    # display
    game.add_game_args(f'+name {name}')
    game.set_window_visible(window_visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_console_enabled(False)
    return game


def _apply_game_wrappers(game, log_rewards):
    variables = {
        vzd.GameVariable.HEALTH: slice(0, 100),
        vzd.GameVariable.AMMO4: slice(0, 50),
        vzd.GameVariable.ARMOR: slice(0, 200),
    }

    game = PreprocessGameWrapper(
        game,
        screen_size=(40, 60),
        labels=[],
        variables=variables,
        depth=False
    )
    game = ModifyButtonsWrapper(
        game,
        digital_buttons=[
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


game = setup_game(name='Lukasz', window_visible=True)

game.add_game_args(
    "-host 1 "
    # This machine will function as a host for a multiplayer game with this many players (including this machine).
    # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
    # "-port 5029 "  # Specifies the port (default is 5029).
    # "+viz_connect_timeout 60 "  # Specifies the time (in seconds), that the host will wait for other players (default is 60).
    "-deathmatch "  # Deathmatch rules are used for the game.
    "+timelimit 2.0 "  # The game (episode) will end after this many minutes have elapsed.
    "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 0 "  # Sets delay between respawns (in seconds, default is 0).
    "+viz_nocheat 1"
)

agent = DQNAgent.load_from_checkpoint('../best_checkpoints/dqn1.ckpt')

game.init()

episodes = 2000

iteration = 0
for i in range(18, episodes, 1):
    print("Episode #" + str(i + 1))

    while not game.is_episode_finished():

        iteration += 1
        # Get the state.
        game_state = game.get_state()

        # Analyze the state.
        action = agent.get_action(game_state, epsilon=0.05)

        # Make your action.
        game.make_action(action, 3)

        if game.is_episode_finished():
            break

        # Check if player is dead
        if game.is_player_dead():
            print("Player died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    # agent.save(f'saved_e{i}')
    print("Episode finished.")
    print("************************")

    print("Results:")
    server_state = game.get_server_state()
    for i in range(len(server_state.players_in_game)):
        if server_state.players_in_game[i]:
            print(
                server_state.players_names[i]
                + ": "
                + str(server_state.players_frags[i])
            )
    print("************************")

    # Starts a new episode. All players have to call new_episode() in multiplayer mode.
    game.new_episode()

game.close()
