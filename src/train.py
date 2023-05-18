import warnings
from pathlib import Path

import vizdoom as vzd
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from lib.agent import RandomAgent
from lib.multi import Player, MultiplayerDoomWrapper, Bot
from lib.dqn import DQNAgent,  DQNPreprocessGameWrapper
from lib.reward import RewardsDoomWrapper, Rewards


def create_agent():
    return DQNAgent(
        n_actions=10,
        epsilon=0.6,
        populate_steps=100,
        batch_size=256
    )
    
    
def create_multiplayer_game():
    player1_agent = RandomAgent(n_actions=10)
    player1_game = create_game(name='player1', window_visible=False)
    player1 = Player(player1_agent, player1_game)
    
    player2_agent = RandomAgent(n_actions=10)
    player2_game = create_game(name='player2')
    player2 = Player(player2_agent, player2_game)
    
    players = [player1, player2]
    bots = [Bot(), Bot(), Bot()]
    
    game = apply_game_wrappers(create_game(name='AI', window_visible=True))
    multiplayer_game = MultiplayerDoomWrapper(game, players, bots)
    return multiplayer_game


def create_game(name, window_visible=False):
    game = vzd.DoomGame()
    
    game.load_config(str(Path(__file__).parent.parent / 'scenarios' / 'cig.cfg'))
    game.set_mode(vzd.Mode.PLAYER)
    game.add_game_args('-deathmatch')
    game.set_episode_timeout(5000)
    
    game.add_game_args(f'+name {name}')
    game.set_window_visible(window_visible)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_console_enabled(False)
    
    return game


def apply_game_wrappers(game):
    rewards = Rewards(
        kill_reward=100,
        death_penalty=100,
        suicide_penalty=200,
        damage_reward=1,
        damage_penalty=1,
        health_reward=1,
        armor_reward=1,
        attack_penalty=1
    )
    game = RewardsDoomWrapper(game, rewards)
    
    game = DQNPreprocessGameWrapper(game)
    
    return game

def create_trainer():
    logger = TensorBoardLogger(
        save_dir=Path(__file__).parent.parent,
        name='logs'
    )

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=-1,
        enable_progress_bar=True,
        logger = logger
    )
    
    return trainer


warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")

agent = create_agent()
game = create_multiplayer_game()
agent.set_train_environment(game)
trainer = create_trainer()
trainer.fit(agent)
