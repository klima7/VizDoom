from abc import ABC
from dataclasses import dataclass

import vizdoom as vzd


class VarTracker(ABC):
    
    def __init__(self, game, var):
        self._game = game
        self._var = var
        
        self._last_reward = 0
        self._total_reward = 0
        
        self._positive_count = 0
        self._negative_count = 0
        
    def reset(self):
        self._last_reward = 0
        self._total_reward = 0
        self._positive_count = 0
        self._negative_count = 0
    
    def update(self, respawned=False):
        pass
    
    def get_name(self):
        return self._var
    
    def get_last_reward(self):
        return self._last_reward
        
    def get_total_reward(self):
        return self._total_reward
    
    def get_positive_count(self):
        return self._positive_count
    
    def get_negative_count(self):
        return self._negative_count
    
    def get_total_count(self):
        return self._positive_count + self._negative_count
    
    def _get_variable_value(self):
        return float(self._game.get_game_variable(self._var))


class NumVarTracker(VarTracker):
    
        def __init__(self, game, name, decrease_reward, increase_reward):
            super().__init__(game, name)
            self.__decrease_reward = decrease_reward
            self.__increase_reward = increase_reward
            self.__last_value = 0
            
        def get_name(self):
            return f'num_{self._var}'
        
        def reset(self):
            super().reset()
            self.__last_value = self._get_variable_value()
            
        def update(self, respawned=False):
            if respawned:
                self.__last_value = self._get_variable_value()
                return

            current_value = self._get_variable_value()
            diff = current_value - self.__last_value
            
            multiplier = self.__increase_reward if diff > 0 else self.__decrease_reward
            self._last_reward = abs(diff) * multiplier
            
            if diff > 0:
                self._positive_count += abs(diff)
            else:
                self._negative_count += abs(diff)
            
            self.__last_value = current_value
            self._total_reward += self._last_reward
        

class BoolVarTracker(VarTracker):
    
    def __init__(self, game, name, true_reward, false_reward):
        super().__init__(game, name)
        self.__true_reward = true_reward
        self.__false_reward = false_reward
        
    def get_name(self):
        return f'bool_{self._var}'
        
    def update(self, death=False):
        value = self._get_variable_value()
        self._last_reward = self.__true_reward if value else self.__false_reward
        self._total_reward += self._last_reward
        
        if value:
            self._positive_count += 1
        else:
            self._negative_count += 1
    
    
class VarTrackersGroup:
    
    def __init__(self, var_rewards):
         self.var_rewards = var_rewards    
         
    def reset(self):
        for var_reward in self.var_rewards:
            var_reward.reset()
         
    def update(self, respawned=False):
        for var_reward in self.var_rewards:
            var_reward.update(respawned)
            
    def get_last_reward(self):
        last_rewards = [var_reward.get_last_reward() for var_reward in self.var_rewards]
        return sum(last_rewards)
    
    def get_last_reward_dict(self):
        last_rewards = {var_reward.get_name(): var_reward.get_last_reward() for var_reward in self.var_rewards}
        return last_rewards
    
    def get_total_reward(self):
        total_rewards = [var_reward.get_total_reward() for var_reward in self.var_rewards]
        return sum(total_rewards)
    
    
@dataclass(kw_only=True)
class Rewards:
    life_reward: float = 0
    kill_reward: float = 0
    hit_reward: float = 0
    damage_reward: float = 0
    health_reward: float = 0
    armor_reward: float = 0
    item_reward: float = 0
    secret_reward: float = 0
    ammo_reward: float = 0
    death_penalty: float = 0
    single_death_penalty: float = 0
    suicide_penalty: float = 0
    hit_penalty: float = 0
    damage_penalty: float = 0
    health_penalty: float = 0
    armor_penalty: float = 0
    ammo_penalty: float = 0
    attack_not_ready_penalty: float = 0
        
    
class RewardsDoomWrapper:
    
    def __init__(self, game, rewards=Rewards(), log=False):
        super().__init__()
        
        self.game = game
        
        self.__damagecount_tracker = NumVarTracker(self, vzd.GameVariable.DAMAGECOUNT, 0, rewards.damage_reward)
        self.__damage_taken_tracker = NumVarTracker(self, vzd.GameVariable.DAMAGE_TAKEN, 0, -rewards.damage_penalty)
        self.__hit_count_tracker = NumVarTracker(self, vzd.GameVariable.HITCOUNT, 0, rewards.hit_reward)
        self.__hits_taken_tracker = NumVarTracker(self, vzd.GameVariable.HITS_TAKEN, 0, -rewards.hit_penalty)
        self.__fragcount_tracker = NumVarTracker(self, vzd.GameVariable.FRAGCOUNT, -rewards.suicide_penalty, rewards.kill_reward)
        self.__health_tracker = NumVarTracker(self, vzd.GameVariable.HEALTH, -rewards.health_penalty, rewards.health_reward)
        self.__armor_tracker = NumVarTracker(self, vzd.GameVariable.ARMOR, -rewards.armor_penalty, rewards.armor_reward)
        self.__deathcount_tracker = NumVarTracker(self, vzd.GameVariable.DEATHCOUNT, 0, -rewards.single_death_penalty)
        self.__itemcount_tracker = NumVarTracker(self, vzd.GameVariable.ITEMCOUNT, 0, rewards.item_reward)
        self.__secretcount_tracker = NumVarTracker(self, vzd.GameVariable.SECRETCOUNT, 0, rewards.secret_reward)
        self.__dead_tracker = BoolVarTracker(self, vzd.GameVariable.DEAD, -rewards.death_penalty, rewards.life_reward)
        self.__attack_ready_tracker = BoolVarTracker(self, vzd.GameVariable.ATTACK_READY, 0, -rewards.attack_not_ready_penalty)
        self.__altattack_ready_tracker = BoolVarTracker(self, vzd.GameVariable.ALTATTACK_READY, 0, 0)
        
        ammo_variables = [vzd.GameVariable.AMMO0, vzd.GameVariable.AMMO1, vzd.GameVariable.AMMO2, 
                          vzd.GameVariable.AMMO3, vzd.GameVariable.AMMO4, vzd.GameVariable.AMMO5,
                          vzd.GameVariable.AMMO6, vzd.GameVariable.AMMO7, vzd.GameVariable.AMMO8,
                          vzd.GameVariable.AMMO0]
        self.__ammo_trackers = [
            NumVarTracker(self, ammo_variable, -rewards.ammo_penalty, rewards.ammo_reward)
            for ammo_variable in ammo_variables
        ]

        self.__trackers = VarTrackersGroup([
            self.__damagecount_tracker,
            self.__damage_taken_tracker,
            self.__hit_count_tracker,
            self.__hits_taken_tracker,
            self.__fragcount_tracker,
            self.__health_tracker,
            self.__armor_tracker,
            self.__deathcount_tracker,
            self.__itemcount_tracker,
            self.__secretcount_tracker,
            self.__dead_tracker,
            self.__attack_ready_tracker,
            self.__altattack_ready_tracker,
            *self.__ammo_trackers
        ])

        self.__log = log

        self.__last_dead = False
        
    def __getattr__(self, attr):
        return getattr(self.game, attr)

    def init(self):
        self.game.init()
        self.__trackers.reset()
        self.__last_dead = False

    def new_episode(self, *args, **kwargs):
        self.game.new_episode(*args, **kwargs)
        self.__last_dead = False
        self.__trackers.reset()
    
    def make_action(self, action, skip=1):
        self.game.make_action(action, skip)
        self.__refresh_reward()
        
    def advance_action(self, tics=1, update_state=True):
        self.game.advance_action(tics, update_state)
        self.__refresh_reward()

    def get_last_reward(self):
        return self.__trackers.get_last_reward()
        
    def get_total_reward(self):
        return self.__trackers.get_total_reward()
        
    def get_frags_count(self):
        return self.__fragcount_tracker.get_positive_count()
        
    def get_suicides_count(self):
        return self.__fragcount_tracker.get_negative_count()
        
    def get_deaths_count(self):
        return self.__deathcount_tracker.get_positive_count()
        
    def get_hits_made_count(self):
        return self.__hit_count_tracker.get_positive_count()
        
    def get_hits_taken_count(self):
        return self.__hits_taken_tracker.get_positive_count()
        
    def get_items_collected_count(self):
        return self.__itemcount_tracker.get_positive_count()
        
    def get_damage_make_count(self):
        return self.__damagecount_tracker.get_positive_count()
        
    def get_damage_taken_count(self):
        return self.__damage_taken_tracker.get_positive_count()
    
    def get_secrets_count(self):
        return self.__secretcount_tracker.get_positive_count()
    
    def get_armor_gained_count(self):
        return self.__armor_tracker.get_positive_count()
    
    def get_armor_lost_count(self):
        return self.__armor_tracker.get_negative_count()
    
    def get_health_gained_count(self):
        return self.__health_tracker.get_positive_count()
    
    def get_health_lost_count(self):
        return self.__health_tracker.get_negative_count()
    
    def get_death_tics_count(self):
        return self.__dead_tracker.get_positive_count()
    
    def get_life_tics_count(self):
        return self.__dead_tracker.get_negative_count()
    
    def get_attack_ready_tics_count(self):
        return self.__attack_ready_tracker.get_positive_count()
    
    def get_attack_not_ready_tics_count(self):
        return self.__attack_ready_tracker.get_negative_count()
    
    def get_altattack_ready_tics_count(self):
        return self.__altattack_ready_tracker.get_positive_count()
    
    def get_altattack_not_ready_tics_count(self):
        return self.__altattack_ready_tracker.get_negative_count()

    def get_ammo_shot_count(self):
        return sum([tracker.get_negative_count() for tracker in self.__ammo_trackers])

    def get_ammo_collected_count(self):
        return sum([tracker.get_positive_count() for tracker in self.__ammo_trackers])

    def get_metrics(self, prefix=''):
        return {
            f'{prefix}total_reward': float(self.get_total_reward()),
            f'{prefix}frags_count': float(self.get_frags_count()),
            # f'{prefix}suicides_count': float(self.get_suicides_count()),
            f'{prefix}deaths_count': float(self.get_deaths_count()),
            f'{prefix}hits_made_count': float(self.get_hits_made_count()),
            f'{prefix}hits_taken_count': float(self.get_hits_taken_count()),
            # f'{prefix}items_collected_count': float(self.get_items_collected_count()),
            f'{prefix}damage_make_count': float(self.get_damage_make_count()),
            f'{prefix}damage_taken_count': float(self.get_damage_taken_count()),
            f'{prefix}armor_gained_count': float(self.get_armor_gained_count()),
            # f'{prefix}armor_lost_count': float(self.get_armor_lost_count()),
            f'{prefix}health_gained_count': float(self.get_health_gained_count()),
            # f'{prefix}health_lost_count': float(self.get_health_lost_count()),
            # f'{prefix}death_tics_count': float(self.get_death_tics_count()),
            f'{prefix}attack_not_ready_tics': float(self.get_attack_not_ready_tics_count()),
            f'{prefix}ammo_shot_count': float(self.get_ammo_shot_count()),
            # f'{prefix}ammo_collected_count': float(self.get_ammo_collected_count()),
        }

    def __refresh_reward(self):
        is_dead = bool(self.game.get_game_variable(vzd.GameVariable.DEAD))
        respawned = self.__last_dead and not is_dead
        self.__last_dead = is_dead
        self.__trackers.update(respawned=respawned)
        if self.__log:
            self.__log_last_rewards()

    def __log_last_rewards(self):
        rewards_dict = self.__trackers.get_last_reward_dict()
        rewards_dict = {name: reward for name, reward in rewards_dict.items() if reward != 0}
        if rewards_dict:
            print('Rewards:', rewards_dict)
