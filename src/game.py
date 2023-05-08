import vizdoom as vzd

class EnhancedDoomGame(vzd.DoomGame):
    
    def __init__(self, kill_reward=10, suicide_penalty=50):
        super().__init__()
        self.__kill_reward = kill_reward
        self.__suicide_penalty = suicide_penalty
        
        self.__player_number = 0
        self.__acknowledge_tic = 0
        
        self.__past_kill_reward = 0
        self.__past_suicide_penalty = 0
        
    def init(self):
        super().init()
        self.__player_number = int(self.get_game_variable(vzd.GameVariable.PLAYER_NUMBER))

    def set_kill_reward(self, reward):
        self.__kill_reward = reward
    
    def get_kill_reward(self):
        return self.__kill_reward
    
    def set_suicide_penalty(self, penalty):
        self.__suicide_penalty = penalty
        
    def get_suicide_penalty(self):
        return self.__suicide_penalty
    
    def get_last_reward(self):
        return super().get_last_reward() \
            + self.__get_last_kill_reward() \
            - self.__get_last_suicide_penalty()
        
    def get_total_reward(self):
        return super().get_total_reward() \
            + self.__past_kill_reward + self.__get_last_kill_reward() \
            - self.__past_suicide_penalty - self.__get_last_suicide_penalty()
    
    def acknowledge(self):
        self.__past_kill_reward += self.__get_last_kill_reward()
        self.__past_suicide_penalty += self.__get_last_suicide_penalty()
        self.__acknowledge_tic = self.get_server_state().tic

    def __get_last_kill_reward(self):
        return self.__kill_reward if self.__has_player_killed_someone() else 0

    def __get_last_suicide_penalty(self):
        return self.__suicide_penalty if self.__has_player_suicide() else 0

    def __has_player_killed_someone(self):
        last_kill_tic = self.get_server_state().players_last_kill_tic[self.__player_number]
        if last_kill_tic >= self.__acknowledge_tic:
            return True
        return False
    
    def __has_player_suicide(self):
        return False
        

doomGame = EnhancedDoomGame()
