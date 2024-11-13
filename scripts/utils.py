# script/utils.py
import yaml

def load_config(config_path="config.yml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_param(config, key, default_value):
    if key not in config:
        print(f"Warning: '{key}' not found in config.yml. Using default value: {default_value}")
    return config.get(key, default_value)