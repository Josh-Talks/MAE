import math
from pydantic import BaseModel
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from typing import List, Literal, Optional, Tuple, Union
import wandb

from MAE.loss import MSELossPatched
from MAE.model import MaskedAutoencoder
from MAE.lr_sched import adjust_learning_rate
from MAE.logging import get_current_lr, get_logger, RunningAverage

logger = get_logger("MAETrainer")


class LRParams(BaseModel):
    lr: float
    sched_type: Literal["step", "cosine"]


class LoggingParams(BaseModel):
    log_after_n_iters: int


class TrainingParameters(BaseModel):
    max_num_epochs: Optional[int]
    max_num_iterations: Optional[int]
    warmup_epochs: int
    current_epoch: int
    current_iteration: int
    lr_params: LRParams
    logging_params: LoggingParams


class MAETrainer:
    def __init__(
        self,
        model: MaskedAutoencoder,
        dataloader: DataLoader[torch.Tensor],
        loss_criteria: MSELossPatched,
        eval_criteria: nn.CrossEntropyLoss,
        scaler: GradScaler,
        optimizer: torch.optim.Optimizer,
        training_params: TrainingParameters,
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.loss_criteria = loss_criteria
        self.eval_criteria = eval_criteria
        self.scaler = scaler
        self.optimizer = optimizer

        self.lr_params = training_params.lr_params
        self.lg_params = training_params.logging_params

        self.current_iteration = training_params.current_iteration
        self.current_epoch = training_params.current_epoch
        self.warmup_epochs = training_params.warmup_epochs

        if training_params.max_num_epochs is not None:
            self.max_num_epochs = training_params.max_num_epochs
            self.max_num_iterations = len(dataloader) * self.max_num_epochs
        elif training_params.max_num_iterations is not None:
            self.max_num_iterations = training_params.max_num_iterations
            self.max_num_epochs = math.ceil(self.max_num_iterations / len(dataloader))
        else:
            raise ValueError("Either max_num_epochs or max_num_iterations must be set.")

        self.scaler = GradScaler()

    def fit(self):
        for _ in range(self.current_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info("Stopping criterion is satisfied. Finishing training")
                return

            self.current_epoch += 1
        logger.info(
            f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training..."
        )

    def train(
        self,
    ):
        train_losses = RunningAverage()

        _ = self.model.train()
        n_batches = len(self.dataloader)
        lr = get_current_lr(self.optimizer)
        wandb.log({"learning-rate": lr}, step=self.current_iteration)

        for x, y in self.dataloader:
            # adjust learning rate
            lr = adjust_learning_rate(
                self.lr_params.sched_type,
                self.lr_params.lr,
                self.max_num_epochs,
                self.warmup_epochs,
                self.current_epoch,
                self.current_iteration,
                self.optimizer,
                n_batches,
            )

            wandb.log(
                {"learning_rate": lr},
                step=self.current_iteration,
            )

            x = x.to(self.model.device)
            y = y.to(self.model.device)

            with autocast("cuda"):
                pred, mask = self.model(x)
                loss = self.loss_criteria(pred, y, mask, self.model)

            train_losses.update(loss.item(), self._batch_size(x))

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            _ = self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.current_iteration % self.lg_params.log_after_n_iters == 0:
                logger.info(f"Training stats. Loss: {train_losses.avg}")

                wandb.log(
                    {
                        "train_loss_avg": train_losses.avg,
                    },
                    step=self.current_iteration,
                )

            if self.max_num_iterations < self.current_iteration:
                return True

            self.current_iteration += 1

        return False

    @staticmethod
    def _batch_size(
        input: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> int:
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
