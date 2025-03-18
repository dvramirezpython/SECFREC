# constants.py

import yaml
import sys
import os

def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the configuration file: {e}")

CONFIG = load_config()
