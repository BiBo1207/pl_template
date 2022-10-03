import torch.nn as nn


def get_activation(act_name: str) -> nn.Module:
    if act_name.upper() == 'RELU':
        return nn.ReLU(inplace=True)
    elif act_name.upper() == 'GELU':
        return nn.GELU()
    elif act_name.upper() == 'SIGMOID':
        return nn.Sigmoid()
    else:
        raise RuntimeError(f'The {act_name} function not exist')