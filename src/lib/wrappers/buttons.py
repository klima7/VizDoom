class ModifyButtonsWrapper:

    def __init__(self, game, available_buttons=(), delta_buttons=None):
        self.game = game
        self.__available_buttons = available_buttons
        self.__delta_buttons = delta_buttons if delta_buttons else {}

        self.__in_buttons = None
        self.__mapping = None

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__in_buttons = self.game.get_available_buttons()
        self.__mapping = self.__create_buttons_mapping()
        print(self.__mapping)
        assert False

    def get_available_buttons(self):
        return tuple(button for button in self.game.get_available_buttons() if button in self.__available_buttons)

    def get_available_buttons_size(self):
        return len(self.__available_buttons)

    def make_action(self, action, *args, **kwargs):
        assert 0 <= action < len(self.__available_buttons), 'Illegal action'
        new_action = self.__mapping[action]
        self.game.make_action(new_action, *args, **kwargs)

    def __create_buttons_mapping(self):
        return {idx: self.__create_vector_for_button(button) for idx, button in enumerate(self.__available_buttons)}

    def __create_vector_for_button(self, button):
        in_button_idx = self.__in_buttons.index(button)
        max_vector = self.__create_max_vector()
        vector = [(val if idx == in_button_idx else 0) for idx, val in enumerate(max_vector)]
        return vector

    def __create_max_vector(self):
        max_vector = [1] * len(self.__in_buttons)
        for button, value in self.__delta_buttons.items():
            idx = self.__in_buttons.index(button)
            max_vector[idx] = value
        return max_vector
