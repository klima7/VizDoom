import numpy as np
import cv2


class PreprocessGameWrapper:

    def __init__(self, game, screen_size, labels=None, variables=None, collect_labels=False, depth=False):
        self.game = game
        self.__screen_size = screen_size
        self.__labels = labels or []
        self.__variables = variables or {}
        self.__collect_labels = collect_labels
        self.__depth = depth

        self.__seen_labels = {}
        self.__available_variables = None

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__available_variables = self.game.get_available_game_variables()

    def get_screen_size(self):
        channels = 1 + int(self.__depth) + len(self.__labels)
        return channels, self.__screen_size[1], self.__screen_size[0]

    def get_variables_size(self):
        return len(self.__variables)

    def get_state(self):
        old_state = self.game.get_state()

        if self.__collect_labels:
            self.__update_seen_labels(old_state)

        if self.game.is_episode_finished():
            screen = np.zeros(self.get_screen_size(), dtype=np.uint8)
            variables = np.zeros(self.get_variables_size(), dtype=np.float32)
        else:
            screen = self.__get_screen(old_state)
            variables = self.__get_variables(old_state.game_variables)

        return {
            'screen': screen,
            'variables': variables
        }

    def __get_screen(self, state):
        parts = []

        screen_buffer = cv2.resize(state.screen_buffer, self.__screen_size)[np.newaxis, ...]
        parts.append(screen_buffer)

        if self.__depth:
            depth_buffer = cv2.resize(state.depth_buffer, self.__screen_size)[np.newaxis, ...]
            parts.append(depth_buffer)

        if self.__labels:
            labels = self.__get_labels(cv2.resize(state.labels_buffer, self.__screen_size, cv2.INTER_NEAREST), state.labels)
            parts.append(labels)

        return np.concatenate(parts, axis=0)

    def __get_variables(self, variables):
        values = []
        for variable, bounds in self.__variables.items():
            assert variable in self.__available_variables, f'Variable {variable} is not available'
            idx = self.__available_variables.index(variable)
            value = float(variables[idx])
            value = min(max(value, bounds.start), bounds.stop)
            value = (value - bounds.start) / (bounds.stop - bounds.start)
            values.append(value)
        return np.array(values, dtype=np.float32)

    def __get_labels(self, labels_buffer, labels):
        maps = []
        for label_name in self.__labels:
            matching_label = [label for label in labels if label.object_name == label_name]
            label_value = matching_label[0].value if matching_label else -1
            map_ = labels_buffer == label_value
            maps.append(map_)
        return np.array(maps, dtype=np.float32)

    def __update_seen_labels(self, state):
        if state:
            for label in state.labels:
                if label.object_name not in self.__seen_labels:
                    self.__seen_labels[label.object_name] = label.value
        print(f'labels ({len(self.__seen_labels)})', self.__seen_labels)
