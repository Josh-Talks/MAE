import math
import torch

from typing import Literal


def step_learning_rate(
    lr: float,
    max_epochs: int,
    warmup_epochs: int,
    epoch: int,
    batch_iter: int,
    optimizer: torch.optim.Optimizer,
    iter_per_epoch: int,
):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch <= warmup_epochs:
        lr_adj = (batch_iter + 1) / (warmup_epochs * iter_per_epoch)
    elif epoch < int(0.3 * max_epochs):
        lr_adj = 1.0
    elif epoch < int(0.6 * max_epochs):
        lr_adj = 1e-1
    elif epoch < int(0.8 * max_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * lr_adj
    return lr * lr_adj


def cosine_learning_rate(
    lr: float,
    max_epochs: int,
    warmup_epochs: int,
    epoch: int,
    batch_iter: int,
    optimizer: torch.optim.Optimizer,
    iter_per_epoch: int,
):
    """Cosine Learning rate"""
    total_epochs = max_epochs
    warm_epochs = warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * iter_per_epoch) + 1e-6
    else:
        lr_adj = (
            1
            / 2
            * (
                1
                + math.cos(
                    batch_iter
                    * math.pi
                    / ((total_epochs - warm_epochs) * iter_per_epoch)
                )
            )
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * lr_adj
    return lr * lr_adj


def adjust_learning_rate(
    lr_sched_type: Literal["cosine", "step"],
    lr: float,
    max_epochs: int,
    warmup_epochs: int,
    epoch: int,
    batch_iter: int,
    optimizer: torch.optim.Optimizer,
    iter_per_epoch: int,
) -> float:
    """Directly adjust the learning rate to new_lr"""
    if lr_sched_type == "cosine":
        # cosine learning rate
        lr = cosine_learning_rate(
            lr,
            max_epochs,
            warmup_epochs,
            epoch,
            batch_iter,
            optimizer,
            iter_per_epoch,
        )
    elif lr_sched_type == "step":
        # step learning rate
        lr = step_learning_rate(
            lr,
            max_epochs,
            warmup_epochs,
            epoch,
            batch_iter,
            optimizer,
            iter_per_epoch,
        )
    else:
        raise ValueError(f"Unknown learning rate scheduler type: {lr_sched_type}")

    return lr
