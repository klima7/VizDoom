class ModifyButtonsWrapper:

    def __init__(self, game, digital_buttons=None, delta_buttons=None):
        digital_buttons = digital_buttons if digital_buttons else []
        delta_buttons = delta_buttons if delta_buttons else {}

        self.game = game
        self.__digital_buttons = digital_buttons
        self.__delta_buttons = delta_buttons
        self.__output_buttons = [*self.__digital_buttons, *self.__delta_buttons.keys()]

        self.__inside_buttons = None
        self.__outside_inside_map = None

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__inside_buttons = self.game.get_available_buttons()
        self.__outside_inside_map = self.__create_outside_inside_map()

    def get_available_buttons(self):
        return tuple(button for button in self.game.get_available_buttons() if button in self.__output_buttons)

    def get_available_buttons_size(self):
        return len(self.__output_buttons)

    def make_action(self, action, *args, **kwargs):
        assert 0 <= action < len(self.__output_buttons), 'Illegal action'
        inside_button = self.__outside_inside_map[action]
        self.game.make_action(inside_button, *args, **kwargs)

    def __create_outside_inside_map(self):
        out_in_map = {}

        for button in self.__digital_buttons:
            out_in_map[len(out_in_map)] = self.__create_inside_digital_action(button)

        for button in self.__delta_buttons:
            in_action_1, in_action_2 = self.__create_inside_delta_actions(button)
            out_in_map[len(out_in_map)] = in_action_1
            out_in_map[len(out_in_map)] = in_action_2

        return out_in_map

    def __create_inside_digital_action(self, outside_button):
        if outside_button is None:
            return [0] * len(self.__inside_buttons)

        inside_button_idx = self.__inside_buttons.index(outside_button)
        vector = [(1 if idx == inside_button_idx else 0) for idx in range(len(self.__inside_buttons))]
        return vector

    def __create_inside_delta_actions(self, outside_button):
        neg_value, pos_value = self.__delta_buttons[outside_button]
        inside_button_idx = self.__inside_buttons.index(outside_button)

        action = [0] * len(self.__inside_buttons)

        neg_inside_action = list(action)
        neg_inside_action[inside_button_idx] = neg_value

        pos_inside_action = list(action)
        pos_inside_action[inside_button_idx] = pos_value

        return neg_inside_action, pos_inside_action
