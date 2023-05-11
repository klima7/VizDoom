import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import cv2

from agent import Agent
from replay import ReplayBuffer, ReplayDataset

import vizdoom as vzd
import matplotlib.pyplot as plt


class PreprocessStateGameWrapper:
    
    def __init__(self, game):
        self.game = game
        
    def __getattr__(self, attr):
        if attr == 'game':
            return self.game
        return getattr(self.game, attr)
    
    def get_state(self):
        state = self.game.get_state()
        return state.screen_buffer
    
    
game = PreprocessStateGameWrapper(vzd.DoomGame())
game.set_mode(vzd.Mode.PLAYER)
game.set_screen_format(vzd.ScreenFormat.GRAY8)
game.init()

state = game.get_state()
cv2.imshow('state', state)
cv2.waitKey(2500000)
