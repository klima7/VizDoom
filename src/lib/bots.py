class AddBotsDoomWrapper:

    def __init__(self, game, bots_count):
        self.game = game
        self.__bots_count = bots_count

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__add_bots()

    def new_episode(self):
        self.game.new_episode()
        self.__add_bots()

    def __add_bots(self):
        for _ in range(self.__bots_count):
            self.send_game_command('addbot')
