import math
from pydantic import BaseModel
import torch
import os
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from typing import List, Literal, Optional, Tuple, Union
from tqdm import tqdm
import wandb

from .logging import get_current_lr, get_logger, RunningAverage, WandbConfig
from .loss import MSELossPatched
from .lr_sched import adjust_learning_rate
from .model import MaskedAutoencoder, ModelConfig
from .utils import save_checkpoint, unpatchify

from MAE.datasets.dataset import get_pretrain_loader, LoaderConfig

logger = get_logger("MAETrainer")


class LoggingParams(BaseModel):
    log_after_n_iters: int
    log_images_after_n_iters: int
    save_nth_ckpt: int
    ckpt_dir: str


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
    device: str = "cuda"
    logging_params: LoggingParams


class MAETrainingConfig(BaseModel):
    logging: WandbConfig
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
        self.device = training_params.device

        self.opt_params = training_params.optimizer_params
        self.lg_params = training_params.logging_params
        self.ckpt_dir = self.lg_params.ckpt_dir

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
        _ = self.model.to(self.device)
        for _ in range(self.current_epoch, self.max_num_epochs):
            logger.info(f"Current epoch {self.current_epoch}/{self.max_num_epochs}")
            # train for one epoch
            loss, should_terminate = self.train()

            if self.current_epoch % self.lg_params.save_nth_ckpt == 0:
                ckpt_name = f"ckpt_epoch_{self.current_epoch}.pytorch"
            else:
                ckpt_name = None

            self._save_checkpoint(loss, ckpt_name)

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
        loss = torch.tensor(float("inf"))

        for x in tqdm(self.dataloader):
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

            x = x.to(self.device)

            with autocast(self.device):
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
                        "train_loss": loss.item(),
                        "train_loss_avg": train_losses.avg,
                    },
                    step=self.current_iteration,
                )

            if self.current_iteration % self.lg_params.log_images_after_n_iters == 0:
                # Unpatchify prediction to get back to original spatial dimensions
                pred_img = unpatchify(
                    pred,
                    self.model.patch_size,
                    self.model.img_size,
                    self.model.in_channels,
                    self.model.spatial_dims,
                )

                # Get first image from batch
                input_img = x[0].detach().cpu().numpy()
                pred_img = pred_img[0].detach().cpu().numpy()

                # Log side-by-side images
                wandb.log(
                    {
                        "input_target": wandb.Image(
                            input_img, caption="Input (Target)"
                        ),
                        "reconstruction": wandb.Image(
                            pred_img, caption="Reconstruction"
                        ),
                    },
                    step=self.current_iteration,
                )

            if self.max_num_iterations < self.current_iteration:
                return float(loss.item()), True

            self.current_iteration += 1

        return float(loss.item()), False

    @staticmethod
    def _batch_size(
        input: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> int:
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def _save_checkpoint(self, loss: float, ckpt_name: Optional[str] = None):

        state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.ckpt_dir, "last_checkpoint.pytorch")
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        save_checkpoint(
            {
                "num_epochs": self.current_epoch + 1,
                "num_iterations": self.current_iteration,
                "model_state_dict": state_dict,
                "loss": loss,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_dir=self.ckpt_dir,
            checkpoint_name=ckpt_name,
        )
