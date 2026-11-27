"""V3 experiment identity over its frozen scientific configuration."""
import hashlib
import json
from pathlib import Path

from ..v2.settings import validate_catalogue

CONFIG = Path(__file__).resolve().parents[2] / 'runs/v3/config.json'
SCHEMA = 'foreground-lesion-multiscale-v3'
DATASET = 'mathewzhou/novo-unet25d-v3-input'
KERNEL = 'mathewzhou/novo-unet25d-v3'


def config():
    return json.loads(CONFIG.read_text())


def science_hash():
    return hashlib.sha256(json.dumps(config(), sort_keys=True).encode()).hexdigest()


def prefix():
    return 'novo-nordisk/experiments/unet25d/v3/' + science_hash()
