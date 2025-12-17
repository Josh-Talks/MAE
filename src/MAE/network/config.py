import yaml
import shutil


def load_config_direct(config_path: str):
    config = yaml.safe_load(open(config_path, "r"))
    return config, config_path


def copy_config(config_path: str, dest_path: str):
    _ = shutil.copyfile(config_path, dest_path)
