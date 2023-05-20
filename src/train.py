import warnings
from pathlib import Path

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from lib.dqn import DQNAgent
from lib.setup import setup_multiplayer_game



agent = DQNAgent(
    n_actions=10,
    epsilon=0.6,
    populate_steps=100,
    batch_size=64,
    update_weights_interval=1_000
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

trainer = Trainer(
    accelerator='cuda',
    max_epochs=-1,
    enable_progress_bar=True,
    logger = logger,
    # profiler=advanced_profiler
)
    
warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")

game = setup_multiplayer_game()
agent.set_train_environment(game)
trainer.fit(agent)
