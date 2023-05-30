import warnings
from pathlib import Path

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from lib.dqn import DQNAgent
from setup import setup_game


warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")

game = setup_game()

agent = DQNAgent(
    n_actions=game.get_available_buttons_size(),
    screen_size=game.get_screen_size(),
    n_variables=game.get_variables_size(),
    lr=0.00025,
    epsilon=0.5,
    populate_steps=1_00,
    buffer_size=80_000,
    batch_size=64,
    actions_per_step=10,
    frames_skip=12,
    weights_update_interval=1_000
)

# agent = DQNAgent.load_from_checkpoint(
#     '/home/klima7/studies/guzw/vizdoom/logs/version_42/checkpoints/epoch=29206-step=508491.ckpt',
#     lr=0.001,
#     epsilon=0.3,
#     buffer_size=15_000
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

trainer = Trainer(
    accelerator='cuda',
    max_epochs=-1,
    enable_progress_bar=True,
    logger=logger,
    # profiler=advanced_profiler
)

agent.set_train_environment(game)
trainer.fit(agent)
