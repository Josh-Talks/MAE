from pathlib import Path
import typer
from typing import Annotated
import wandb

from MAE.network.trainer import create_trainer, MAETrainingConfig
from MAE.network.config import load_config_direct, copy_config


def main(
    config: Annotated[str, typer.Option(help="Path to the config file", exists=True)],
):
    cfg_data, _ = load_config_direct(config)

    cfg = MAETrainingConfig.model_validate(cfg_data)

    _ = wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.name,
        mode=cfg.logging.mode,
        config=cfg_data,
    )

    # Create trainer
    trainer = create_trainer(cfg)

    # Copy config file
    yaml_filename = Path(config).name
    copy_config(
        config, Path(cfg.training_params.logging_params.ckpt_dir) / yaml_filename
    )
    # Start training
    trainer.fit()


if __name__ == "__main__":
    typer.run(main)
