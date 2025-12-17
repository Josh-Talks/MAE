import math
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from typing import List, Literal, Optional, Tuple, Union
import wandb

from .loss import MSELossPatched
from .model import MaskedAutoencoder, ModelConfig
from .lr_sched import adjust_learning_rate
from .logging import get_current_lr, get_logger, RunningAverage

from MAE.datasets.dataset import get_pretrain_loader, LoaderConfig

logger = get_logger("MAETrainer")


class LoggingParams(BaseModel):
    log_after_n_iters: int


class OptimizerParams(BaseModel):
    optimizer_type: Literal["AdamW"]
    weight_decay: float
    lr: float
    lr_sched_type: Literal["step", "cosine"]


class TrainingParameters(BaseModel):
    max_num_epochs: Optional[int]
    max_num_iterations: Optional[int]
    warmup_epochs: int
    current_epoch: int
    current_iteration: int
    optimizer_params: OptimizerParams
    logging_params: LoggingParams


class MAETrainingConfig(BaseModel):
    loss_type: Literal["MSE"]
    optimizer_type: Literal["AdamW"]
    model_params: ModelConfig
    training_params: TrainingParameters
    dataloader: LoaderConfig


def create_trainer(config: MAETrainingConfig):
    # create model
    model_cfg = config.model_params
    model = MaskedAutoencoder(
        img_size=model_cfg.img_size,
        patch_size=model_cfg.patch_size,
        in_channels=model_cfg.in_channels,
        embed_dim=model_cfg.embed_dim,
        num_classes=model_cfg.num_classes,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        decoder_embed_dim=model_cfg.decoder_embed_dim,
        decoder_depth=model_cfg.decoder_depth,
        decoder_num_heads=model_cfg.decoder_num_heads,
        pos_embed_type=model_cfg.pos_embed_type,
        qkv_bias=model_cfg.qkv_bias,
        dropout_rate=model_cfg.dropout_rate,
        attn_drop=model_cfg.attn_drop,
        drop_path=model_cfg.drop_path,
        mask_ratio=model_cfg.mask_ratio,
        mlp_ratio=model_cfg.mlp_ratio,
        spatial_dims=model_cfg.spatial_dims,
    )

    # create loss
    if config.loss_type == "MSE":
        loss_criteria = MSELossPatched()
    else:
        raise ValueError(f"Unsupported loss type: {config.loss_type}")

    if config.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training_params.optimizer_params.lr,
            weight_decay=config.training_params.optimizer_params.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

    pretrain_loader = get_pretrain_loader(config.dataloader)

    return MAETrainer(
        model=model,
        dataloader=pretrain_loader,
        loss_criteria=loss_criteria,
        scaler=GradScaler(),
        optimizer=optimizer,
        training_params=config.training_params,
    )


class MAETrainer:
    def __init__(
        self,
        model: MaskedAutoencoder,
        dataloader: DataLoader[torch.Tensor],
        loss_criteria: MSELossPatched,
        scaler: GradScaler,
        optimizer: torch.optim.Optimizer,
        training_params: TrainingParameters,
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.loss_criteria = loss_criteria
        self.scaler = scaler
        self.optimizer = optimizer

        self.opt_params = training_params.optimizer_params
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

        self.scaler = scaler

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

        for x in self.dataloader:
            # adjust learning rate
            lr = adjust_learning_rate(
                self.opt_params.lr_sched_type,
                self.opt_params.lr,
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

            with autocast("cuda"):
                pred, mask = self.model(x)
                loss = self.loss_criteria(pred, x, mask, self.model)

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
