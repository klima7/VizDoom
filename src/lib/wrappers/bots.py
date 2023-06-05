class AddBotsDoomWrapper:

    def __init__(self, game, bots_count, difficulty=None):
        """
        Valid difficulties: None, 1, 2, 3, 4, 5 (None - default, 1 - easy, 5 - very hard)
        Difficulty works only with unknown_1 and unknown_3 maps
        """
        self.game = game
        self.__bots_count = bots_count
        self.__difficulty = difficulty

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__add_bots()

    def new_episode(self):
        self.game.new_episode()
        self.__add_bots()

    def __add_bots(self):
        self.send_game_command('removebots')

        for _ in range(self.__bots_count):
            self.send_game_command('addbot')

        if self.__difficulty is not None:
            self.game.send_game_command(f'pukename change_difficulty {self.__difficulty}')
