from typing import Dict
from train import train
from ruamel.yaml import YAML


def load_args(path: str = './config.yaml') -> Dict:
    with open(path, 'r') as f:
        data = YAML(typ='safe').load(f)
    return data


if __name__ == '__main__':
    args = load_args()
    train(args)
