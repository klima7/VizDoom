import warnings
from pathlib import Path

from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from lib.dqn import DQNAgent
from lib.setup import setup_multiplayer_game



agent = DQNAgent(
    n_actions=10,
    epsilon=0.6,
    populate_steps=100,
    batch_size=256
)

logger = TensorBoardLogger(
    save_dir=Path(__file__).parent.parent,
    name='logs'
)

trainer = Trainer(
    accelerator='cpu',
    max_epochs=1,
    enable_progress_bar=True,
    logger = logger,
    profiler="simple"
)
    
warnings.filterwarnings("ignore", ".*train_dataloader, does not have many workers.*")

game = setup_multiplayer_game()
agent.set_train_environment(game)
trainer.fit(agent)
