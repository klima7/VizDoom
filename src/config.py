import os

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
