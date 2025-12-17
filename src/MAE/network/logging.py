import logging
from typing_extensions import Literal
from pydantic import BaseModel
import sys
import torch
from typing import Dict, Optional, Union


class WandbConfig(BaseModel):
    project: str
    name: str
    mode: Literal["disabled", "online", "offline"]
    run_id: Optional[str] = None
    resume: Union[bool, None, Literal["allow", "never", "must", "auto"]] = None


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    lrs = [param_group.get("lr", None) for param_group in optimizer.param_groups]
    lrs = [lr for lr in lrs if lr is not None]
    # to keep things simple we only return one of the valid lrs
    return lrs[0]


class RunningAverage:
    """Computes and stores the average"""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value: float, n: int = 1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger
