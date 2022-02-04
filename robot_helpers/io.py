from pathlib import Path
import pickle
import yaml


def load_yaml(path):
    with Path(path).open("r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def load_pickle(path):
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with Path(path).open("wb") as f:
        pickle.dump(data, f)
