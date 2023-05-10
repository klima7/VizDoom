from abc import ABC

import vizdoom as vzd


class _VarReward(ABC):
    
    def __init__(self, game, var_name):
        self._game = game
        self._var_name = var_name
        self._last_reward = 0
        self._total_reward = 0
        
    def reset(self):
        self._last_reward = 0
        self._total_reward = 0
    
    def update(self):
        pass
    
    def get_name(self):
        return self._var_name
    
    def get_last_reward(self):
            return self._last_reward
        
    def get_total_reward(self):
            return self._total_reward
    
    def _get_variable_value(self):
            return float(self._game.get_game_variable(self._var_name))


class _NumVarReward(_VarReward):
    
        def __init__(self, game, name, decrease_reward, increase_reward):
            super().__init__(game, name)
            self.__decrease_reward = decrease_reward
            self.__increase_reward = increase_reward
            self.__last_value = 0
            
        def get_name(self):
            return f'num_{self._var_name}'
        
        def reset(self):
            super().reset()
            self.__last_value = self._get_variable_value()
            
        def update(self):
            current_value = self._get_variable_value()
            diff = current_value - self.__last_value
            multiplier = self.__increase_reward if diff > 0 else self.__decrease_reward
            self._last_reward = abs(diff) * multiplier
            self.__last_value = current_value
            self._total_reward += self._last_reward
        

class _BoolVarReward(_VarReward):
    
    def __init__(self, game, name, true_reward, false_reward):
        super().__init__(game, name)
        self.__true_reward = true_reward
        self.__false_reward = false_reward
        
    def get_name(self):
            return f'bool_{self._var_name}'
        
    def update(self):
        value = self._get_variable_value()
        self._last_reward = self.__true_reward if value else self.__false_reward
        self._total_reward += self._last_reward
    
    
class _VarRewardsGroup:
    
    def __init__(self, var_rewards):
         self.var_rewards = var_rewards    
         
    def reset(self):
        for var_reward in self.var_rewards:
            var_reward.reset()
         
    def update(self):
        for var_reward in self.var_rewards:
            var_reward.update()
            
    def get_last_reward(self):
        last_rewards = [var_reward.get_last_reward() for var_reward in self.var_rewards]
        return sum(last_rewards)
    
    def get_last_reward_dict(self):
        last_rewards = { var_reward.get_name(): var_reward.get_last_reward() for var_reward in self.var_rewards }
        return last_rewards
    
    def get_total_reward(self):
        total_rewards = [var_reward.get_total_reward() for var_reward in self.var_rewards]
        return sum(total_rewards)
        
    
class RewardedDoomGame(vzd.DoomGame):
    
    def __init__(self, rewards_config, log=False):
        super().__init__()
        self.__reward_group = self.__create_rewards(rewards_config)
        self.__log = log

    def new_episode(self, *args, **kwargs):
        super().new_episode(*args, **kwargs)
        self.__reward_group.reset()

    def get_last_reward(self):
        return self.__reward_group.get_last_reward()
        
    def get_total_reward(self):
        return self.__reward_group.get_total_reward()
    
    def make_action(self, action, skip=1):
        super().make_action(action, skip)
        self.__update_reward()
        
    def advance_action(self, tics=1, update_state=True):
        super().advance_action(tics, update_state)
        self.__update_reward()
        
    def __update_reward(self):
        self.__reward_group.update()
        
        if self.__log:
            self.__log_last_rewards()
        
    def __log_last_rewards(self):
        rewards_dict = self.__reward_group.get_last_reward_dict()
        rewards_dict = { name: reward for name, reward in rewards_dict.items() if reward != 0 }
        if rewards_dict:
            print('Rewards:', rewards_dict)
            
    def __create_rewards(self, cfg):
        var_rewards = [
            _NumVarReward(self, vzd.GameVariable.DAMAGECOUNT, 0, cfg.damage_reward),
            _NumVarReward(self, vzd.GameVariable.DAMAGE_TAKEN, 0, -cfg.damage_penalty),
            _NumVarReward(self, vzd.GameVariable.HITCOUNT, 0, cfg.hit_reward),
            _NumVarReward(self, vzd.GameVariable.HITS_TAKEN, 0, -cfg.hit_penalty),
            _NumVarReward(self, vzd.GameVariable.FRAGCOUNT, -cfg.suicide_penalty, cfg.kill_reward),
            _NumVarReward(self, vzd.GameVariable.HEALTH, -cfg.health_penalty, cfg.health_reward),
            _NumVarReward(self, vzd.GameVariable.ARMOR, -cfg.armor_penalty, cfg.armor_reward),
            _NumVarReward(self, vzd.GameVariable.DEATHCOUNT, 0, -cfg.single_death_penalty),
            _NumVarReward(self, vzd.GameVariable.ITEMCOUNT, 0, cfg.item_reward),
            _NumVarReward(self, vzd.GameVariable.SECRETCOUNT, 0, cfg.secret_reward),
            _BoolVarReward(self, vzd.GameVariable.DEAD, cfg.live_reward, -cfg.death_penalty),
            _BoolVarReward(self, vzd.GameVariable.ATTACK_READY, -cfg.attack_penalty, cfg.attack_reward),
            _BoolVarReward(self, vzd.GameVariable.ALTATTACK_READY, -cfg.alt_attack_penalty, cfg.alt_attack_reward),
        ]
        return _VarRewardsGroup(var_rewards)
