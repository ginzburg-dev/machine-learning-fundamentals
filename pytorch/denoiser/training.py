from typing import Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path

from pytorch.denoiser.dataset import (
    random_patch_pair,
)

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset

import structlog

LOGGER = structlog.get_logger("DNN Logger")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    steps_per_batch: int = 200,
    patch_size: int = 64,
    max_alpha_search_tries: int = 50,
) -> float:
    """One epoch train step."""
    model.train()
    num_steps = 0.0
    running_loss = 0.0
    for input, target, alpha in dataloader:
        for _ in range(steps_per_batch):
            input_patch, target_patch, found = random_patch_pair(
                input=input,
                target=target,
                patch_size=patch_size,
                alpha=alpha,
                max_alpha_search_tries=max_alpha_search_tries)
            
            if found is None:
                continue

            input = input_patch.to(device)
            target = target_patch.to(device)

            pred = model(input)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_steps += 1

    return running_loss/max(num_steps, 1)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model."""
    model.eval()
    num_batches = 0.0
    running_loss = 0.0
    for input, target, _ in dataloader:
            input = input.to(device)
            target = target.to(device)

            pred = model(input)
            loss = loss_fn(pred, target)

            running_loss += loss.item()
            num_batches += 1

    return running_loss/max(num_batches, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    cv_loader: DataLoader | None,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    output_weights: str | Path,
    print_every_n_steps: int = 10,
    steps_per_batch: int = 200,
    patch_size: int = 64,
    max_alpha_search_tries: int = 50,
) -> Tuple[dict[str, Any], Any, Any, Any]: 
    """
    Training loop.

    Returns:
        tuple: A tuple containing:
            model_weights: The trained model weights.
            weights_out_file_name: Filename where the weights were saved.
            train_avg_loss: Average training loss.
            cross_validation_avg_loss: Average cross-validation loss.
    """
    output_weights = Path(output_weights)
    output_weights.parent.mkdir(parents=True, exist_ok=True)

    found_best = False
    best_model_state = last_model_state = model.state_dict()

    best_cv_loss: float = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss: float = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            steps_per_batch,
            patch_size,
            max_alpha_search_tries,
        )

        if cv_loader is not None:
            cv_loss = evaluate(model, cv_loader, loss_fn, device)
        else:
            cv_loss = float("nan")

        if cv_loader is not None and cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, output_weights)
            found_best = True

        if (epoch + 1) % print_every_n_steps == 0:
                print(f"Epoch {epoch+1}/{epochs}, train_loss = {train_loss:.6f},",
                    f"cv_loss = {cv_loss:.6f}," if cv_loader is not None else "",
                    f"PROGRESS: {int(((epoch+1)/epochs)*100)}%")

    return_model_state = best_model_state if cv_loader is not None else last_model_state

    if not found_best:
        torch.save(model.state_dict(), output_weights)

    print(f"\nModel was trained successfully!")
    print(f"Model weights saved to {output_weights}")

    return (
        return_model_state,
        output_weights,
        train_loss, # pyright: ignore[reportPossiblyUnboundVariable]
        cv_loss, # pyright: ignore[reportPossiblyUnboundVariable]
    )

