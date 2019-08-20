import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


def make_pool1d_layer(param_dict):
    params = (
        param_dict['kernel'], param_dict.get('stride', param_dict['kernel']),
        param_dict.get('padding', 0)
    )
    if param_dict.get('op', 'avg') == 'max':
        return nn.MaxPool3d(*params)
    else:
        return nn.AvgPool3d(*params)


class Conv1dAutoEncoder(nn.Module):

    def __init__(self, conv_params):
        super(Conv1dAutoEncoder, self).__init__()
        self.conv_params = conv_params
        self.build()

    def build(self):
        encoder_mods = []

        for p in self.conv_params:
            k = p.get('kernel', 3)
            pad = k // 2
            encoder_mods.append(
                nn.Conv1d(
                    p['in'], p['out'], k, p.get('stride', 1), pad,
                    p.get('dilation', 1), p.get('groups', 1),
                    p.get('bias', True)
                )
            )
            encoder_mods.append(nn.BatchNorm1d(p['out']))
            encoder_mods.append(p.get('activation', nn.ReLU)())
            if 'pool' in p:
                encoder_mods.append(make_pool1d_layer(p['pool']))

        # TODO: Add a flatten -> dense layer to do the reduction to 2 dims
        # Easiest to do this with a seperate sequential, that way the
        # corresponding decoder is a bit simpler...

        # package encoding layers as a Sequential network
        self.encoder_net = nn.Sequential(*encoder_mods)

    def encode(self, X):
        return self.encoder_net(X)

    def decode(self, X):
        # TODO: Add the dense dimension expansion -> reshaping
        for layer in reversed(self.encoder_net, self.conv_params):
            pad = layer.weight.shape[-1] // 2
            X = F.conv_transpose1d(
                X, layer.weight, layer.bias, layer.stride, pad, pad,
                layer.groups, layer.dilation
            )
        return X

    def forward(self, X):
        return self.decode(self.encode(X))
