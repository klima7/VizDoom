class ModifyButtonsWrapper:

    def __init__(self, game, available_buttons=None, delta_buttons=None):
        self.game = game
        self.__available_buttons = available_buttons if available_buttons else []
        self.__delta_buttons = delta_buttons if delta_buttons else {}

        self.__inside_buttons = None
        self.__outside_inside_map = None

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__inside_buttons = self.game.get_available_buttons()
        self.__outside_inside_map = self.__create_outside_inside_map()

    def get_available_buttons(self):
        return tuple(button for button in self.game.get_available_buttons() if button in self.__available_buttons)

    def get_available_buttons_size(self):
        return len(self.__available_buttons)

    def make_action(self, action, *args, **kwargs):
        assert 0 <= action < len(self.__available_buttons), 'Illegal action'
        inside_button = self.__outside_inside_map[action]
        self.game.make_action(inside_button, *args, **kwargs)

    def __create_outside_inside_map(self):
        return {idx: self.__create_inside_action(button) for idx, button in enumerate(self.__available_buttons)}

    def __create_inside_action(self, outside_button):
        inside_button_idx = self.__inside_buttons.index(outside_button)
        inside_action = self.__create_max_inside_action()
        vector = [(val if idx == inside_button_idx else 0) for idx, val in enumerate(inside_action)]
        return vector

    def __create_max_inside_action(self):
        max_inside_action = [1] * len(self.__inside_buttons)
        for button, value in self.__delta_buttons.items():
            inside_button_idx = self.__inside_buttons.index(button)
            max_inside_action[inside_button_idx] = value
        return max_inside_action
