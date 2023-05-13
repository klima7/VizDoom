from pathlib import Path

import vizdoom as vzd
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from lib.multi import MultiDoomGame
from lib.configs import GameConfig, RewardsConfig, AgentConfig, BotConfig
from lib.agent import RandomAgent
from lib.dqn_agent import DQNAgent, PreprocessStateGameWrapper


def create_environment():
    rewards_config = RewardsConfig(
        damage_reward=1,
        damage_penalty=1,
        death_penalty=50,
    )

    game_config = GameConfig(
        config_name='cig.cfg',
        timeout=300
    )

    player_config = AgentConfig(
        name='AI',
        rewards_config=rewards_config,
        window_visible=True,
        screen_format=vzd.ScreenFormat.GRAY8,
        screen_resolution=vzd.ScreenResolution.RES_320X240,
    )

    agents_configs = [
        AgentConfig(
            name='oponent1',
            agent=RandomAgent(),
            window_visible=False
        ),
        AgentConfig(
            name='oponent2',
            agent=RandomAgent(),
            window_visible=False
        ),
    ]

    bots_configs = [
        BotConfig(),
        BotConfig(),
        BotConfig(),
    ]

    env = MultiDoomGame(game_config, player_config, agents_configs, bots_configs)
    env = PreprocessStateGameWrapper(env)
    return env


env = create_environment()
n_actions = env.get_available_buttons_size()

agent = DQNAgent(
    n_actions=n_actions,
    epsilon=0.6,
    populate_steps=100,
    batch_size=256
)

agent.set_train_environment(env)

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

trainer.fit(agent)
