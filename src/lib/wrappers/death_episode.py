class FinishEpisodeOnDeathDoomWrapper:

    def __init__(self, game):
        self.game = game

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def is_episode_finished(self):
        return self.game.is_episode_finished() or self.game.is_player_dead()
