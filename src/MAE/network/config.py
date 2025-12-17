from pathlib import Path
import shutil
from typing import Union
import yaml


def load_config_direct(config_path: str):
    config = yaml.safe_load(open(config_path, "r"))
    return config, config_path


def copy_config(config_path: Union[str, Path], dest_path: Union[str, Path]):
    _ = shutil.copyfile(config_path, dest_path)
