class SingleMapDoomWrapper:

    def __init__(self, game, map):
        self.game = game
        self.__map = map

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.set_doom_map(self.__map)

    def new_episode(self):
        self.game.new_episode()
        self.set_doom_map(self.__map)
