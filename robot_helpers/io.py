import yaml
from pathlib import Path


def load_yaml(path):
    with Path(path).open("r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
