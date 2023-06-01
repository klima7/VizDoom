import warnings
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint

from lib.dqn import DQNAgent
from setup import setup_game


warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")

game = setup_game()

agent = DQNAgent(
    n_actions=game.get_available_buttons_size(),
    screen_size=game.get_screen_size(),
    n_variables=game.get_variables_size(),

    lr=0.00025,
    batch_size=256,

    gamma=0.99,
    epsilon=0.7,
    populate_steps=1_000,
    buffer_size=150_000,
    actions_per_step=10,
    frames_skip=3,
    validation_interval=50,
    weights_update_interval=1_000,

    epsilon_update_interval=2_000,
    epsilon_min=0.05,
)

# agent = DQNAgent.load_from_checkpoint(
#     '../logs/version_42/checkpoints/last.ckpt',
# )

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
    filename='best-{val_total_reward:.2f}',
    monitor='val_total_reward',
    mode='max',
    save_last=True,
    save_top_k=5,
    save_on_train_epoch_end=True,
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
