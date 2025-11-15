import torch
from torch.optim import Optimizer, Adam

from dataclasses import dataclass

@dataclass
class TrainConfig():
    epochs: int
    optimizer: Optimizer
    lr: float = 1e-3
