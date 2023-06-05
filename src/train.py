import warnings
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint

from lib.dqn import DQNAgent
from lib.actor_critic import ActorCriticAgent
from setup import setup_game


warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")

ALGORITHM = 'dqn'

game = setup_game()

if ALGORITHM == 'dqn':
    agent = DQNAgent(
        n_actions=game.get_available_buttons_size(),
        screen_size=game.get_screen_size(),
        n_variables=game.get_variables_size(),

        lr=0.00025,
        batch_size=64,

        gamma=0.99,
        epsilon=0.7,
        populate_steps=1_000,
        buffer_size=30_000,
        actions_per_step=5,
        frames_skip=5,
        validation_interval=50,
        weights_update_interval=500,

        epsilon_update_interval=2_000,
        epsilon_min=0.05,
    )

    # agent = DQNAgent.load_from_checkpoint(
    #     '../logs/version_42/checkpoints/last.ckpt',
    # )

elif ALGORITHM == 'actor_critic':
    agent = ActorCriticAgent(
        n_actions=game.get_available_buttons_size(),
        screen_size=game.get_screen_size(),
        n_variables=game.get_variables_size(),
        batch_size=32,
        lr_actor=0.001,
        lr_critic=0.001,
        frames_skip=3,
        gamma=0.99,
        buffer_size=32,
        populate_steps=32,
        actions_per_step=32,
        validation_interval=500000000000000,
    )

logger = TensorBoardLogger(
    save_dir=Path(__file__).parent.parent,
    name='logs'
)

simple_profiler = SimpleProfiler(
    dirpath='../logs',
    filename='simple',
    extended=True,
)

advanced_profiler = AdvancedProfiler(
    dirpath='../logs',
    filename='advanced',
)

checkpoint_checkpoint = ModelCheckpoint(
    save_last=True,
    save_top_k=-1,
    every_n_epochs=50,
)

trainer = Trainer(
    accelerator='cuda',
    max_epochs=-1,
    enable_progress_bar=True,
    logger=logger,
    callbacks=[checkpoint_checkpoint],
    # profiler=advanced_profiler
)

agent.set_train_environment(game)
trainer.fit(agent)
