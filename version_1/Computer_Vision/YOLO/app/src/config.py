import yaml
from pathlib import Path

def load_config(config_path="configs/default.yaml"):
    with open(config_path,"r") as f:
        config=yaml.safe_load(f)
    return config


def save_config(config,config_path="configs/default.yaml"):
    with open(config_path,"r") as f:
        yaml.dump(config,f)