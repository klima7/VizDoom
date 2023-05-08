from abc import ABC

import vizdoom as vzd


class VarReward(ABC):
    
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


class NumVarReward(VarReward):
    
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
        

class BoolVarReward(VarReward):
    
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
    
    
class VarRewardsGroup:
    
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
    
    
class RewardsConfig:
    
    def __init__(
        self,
        live_reward = 0,
        kill_reward = 0,
        hit_reward = 0, 
        damage_reward = 0,
        health_reward = 0,
        armor_reward = 0,
        item_reward = 0,
        secret_reward = 0,
        attack_reward = 0,
        alt_attack_reward = 0,
        death_penalty = 0,
        single_death_penalty = 0,
        suicide_penalty = 0,
        hit_penalty = 0,
        damage_penalty = 0,
        health_penalty = 0,
        armor_penalty = 0,
        attack_penalty = 0,
        alt_attack_penalty = 0,
    ):
        self.live_reward = live_reward
        self.kill_reward = kill_reward
        self.hit_reward = hit_reward 
        self.damage_reward = damage_reward
        self.health_reward = health_reward
        self.armor_reward = armor_reward
        self.item_reward = item_reward
        self.secret_reward = secret_reward
        self.attack_reward = attack_reward
        self.alt_attack_reward = alt_attack_reward
        self.death_penalty = death_penalty
        self.single_death_penalty = single_death_penalty
        self.suicide_penalty = suicide_penalty
        self.hit_penalty = hit_penalty
        self.damage_penalty = damage_penalty
        self.health_penalty = health_penalty
        self.armor_penalty = armor_penalty
        self.attack_penalty = attack_penalty
        self.alt_attack_penalty = alt_attack_penalty
        
    def create_rewards_group(self, game):
        var_rewards = [
            NumVarReward(game, vzd.GameVariable.DAMAGECOUNT, 0, self.damage_reward),
            NumVarReward(game, vzd.GameVariable.DAMAGE_TAKEN, 0, -self.damage_penalty),
            NumVarReward(game, vzd.GameVariable.HITCOUNT, 0, self.hit_reward),
            NumVarReward(game, vzd.GameVariable.HITS_TAKEN, 0, -self.hit_penalty),
            NumVarReward(game, vzd.GameVariable.FRAGCOUNT, -self.suicide_penalty, self.kill_reward),
            NumVarReward(game, vzd.GameVariable.HEALTH, -self.health_penalty, self.health_reward),
            NumVarReward(game, vzd.GameVariable.ARMOR, -self.armor_penalty, self.armor_reward),
            NumVarReward(game, vzd.GameVariable.DEATHCOUNT, 0, -self.single_death_penalty),
            NumVarReward(game, vzd.GameVariable.ITEMCOUNT, 0, self.item_reward),
            NumVarReward(game, vzd.GameVariable.SECRETCOUNT, 0, self.secret_reward),
            BoolVarReward(game, vzd.GameVariable.DEAD, self.live_reward, -self.death_penalty),
            BoolVarReward(game, vzd.GameVariable.ATTACK_READY, -self.attack_penalty, self.attack_reward),
            BoolVarReward(game, vzd.GameVariable.ALTATTACK_READY, -self.alt_attack_penalty, self.alt_attack_reward),
        ]
        return VarRewardsGroup(var_rewards)

    
class RewardedDoomGame(vzd.DoomGame):
    
    def __init__(self, rewards_config, log=False):
        super().__init__()
        self.__reward_group = rewards_config.create_rewards_group(self)
        self.__log = log

    def new_episode(self, *args, **kwargs):
        super().new_episode(*args, **kwargs)
        self.__reward_group.reset()
        
    def update_reward(self):
        self.__reward_group.update()
        
        if self.__log:
            self.__log_last_rewards()

    def get_last_reward(self):
        return self.__reward_group.get_last_reward()
        
    def get_total_reward(self):
        return self.__reward_group.get_total_reward()
        
    def __log_last_rewards(self):
        rewards_dict = self.__reward_group.get_last_reward_dict()
        rewards_dict = { name: reward for name, reward in rewards_dict.items() if reward != 0 }
        if rewards_dict:
            print('Rewards:', rewards_dict)
