from typing import Tuple
from abc import ABC, abstractmethod

from models

import torch
from torch import nn
from torch.optim import Optimizer, Adam


class DNNBaseModel(ABC):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.device = device

    @abstractmethod
    @staticmethod
    def _load_image_tensor(path: str) -> torch.Tensor:
        ...

    @abstractmethod
    @staticmethod
    def save_image(
        tensor: torch.Tensor,
        path: str
    ) -> None:
        ...

    @abstractmethod
    @staticmethod
    def random_patch_pair(
        noisy,
        clean,
        patch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def train(
        self,
        lr: float = 1e-3,
        optimizer: Optimizer | None = None,
        input: list[torch.Tensor] | None = None,
        target: list[torch.Tensor] | None = None
    ) -> None:
        ...

    def apply_mode(args, device) -> None:
