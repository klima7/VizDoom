import numpy as np
import vizdoom as vzd


class DQNPreprocessGameWrapper:
    # BulletPuff, DoomPlayer, TeleportFog, Blood
    IMPORTANT_LABELS = [
        'DoomPlayer',
        'TeleportFog',
    ]

    IMPORTANT_VARIABLES = {
        vzd.GameVariable.HEALTH: slice(0, 200),
        vzd.GameVariable.AMMO4: slice(0, 50),
    }

    def __init__(self, game, collect_labels=False):
        self.game = game
        self.__collect_labels = collect_labels
        self.__seen_labels = {}
        self.__available_variables = None
        self.__tmp = 0

    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__available_variables = self.game.get_available_game_variables()

    def get_state(self):
        state = self.game.get_state()

        self.__tmp += 1
        if self.__collect_labels and self.__tmp > 2000:
            self.__update_seen_labels(state)

        if self.game.is_episode_finished():
            w = self.game.get_screen_width()
            h = self.game.get_screen_height()
            c = self.game.get_screen_channels()

            screen = np.zeros((c + 1 + len(self.IMPORTANT_LABELS), h, w), dtype=np.float32)
            automap = np.zeros((1, h, w), dtype=np.float32)
            variables = np.zeros((len(self.IMPORTANT_VARIABLES, )), dtype=np.float32)

        else:
            screen = self.__get_screen(state)
            automap = self.__get_automap(state.automap_buffer)
            variables = self.__get_important_variables(state.game_variables)

        return {
            'screen': screen,
            'automap': automap,
            'variables': variables
        }

    def __get_screen(self, state):
        screen_buffer = state.screen_buffer[np.newaxis, ...] / 255  # 240x320
        depth_buffer = state.depth_buffer[np.newaxis, ...] / 255
        labels = self.__get_important_labels_map(state.labels_buffer, state.labels)
        screen = np.concatenate([screen_buffer, depth_buffer, labels], axis=0).astype(np.float32)
        return screen

    def __get_automap(self, automap_buffer):
        automap = automap_buffer[np.newaxis, ...] / 255
        automap = automap.astype(np.float32)
        return automap

    def __get_important_variables(self, variables):
        values = []
        for variable, bounds in self.IMPORTANT_VARIABLES.items():
            assert variable in self.__available_variables, f'Variable {variable} is not available'
            idx = self.__available_variables.index(variable)
            value = float(variables[idx])
            value = min(max(value, bounds.start), bounds.stop)
            value = (value - bounds.start) / (bounds.stop - bounds.start)
            values.append(value)
        return np.array(values, dtype=np.float32)

    def __get_important_labels_map(self, labels_buffer, labels):
        maps = []
        for label_name in self.IMPORTANT_LABELS:
            matching_label = [label for label in labels if label.object_name == label_name]
            label_value = matching_label[0].value if matching_label else -1
            map = labels_buffer == label_value
            maps.append(map)
        return np.array(maps, dtype=np.float32)

    def __update_seen_labels(self, state):
        if state:
            for label in state.labels:
                if label.object_name not in self.__seen_labels:
                    self.__seen_labels[label.object_name] = label.value
        print(f'labels ({len(self.__seen_labels)})', self.__seen_labels)
