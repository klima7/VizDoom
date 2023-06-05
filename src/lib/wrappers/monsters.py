class SetMonstersDoomWrapper:

    def __init__(self, game, monsters_count):
        self.game = game
        self.__monsters_count = monsters_count

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.game.send_game_command(f'pukename change_num_of_monster_to_spawn {self.__monsters_count}')
