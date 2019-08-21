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


class Squeezer(nn.Module):
    def __init__(self):
        super(Squeezer, self).__init__()

    @staticmethod
    def forward(X):
        return torch.squeeze(X)


class Unsqueezer(nn.Module):
    def __init__(self, dim):
        super(Unsqueezer, self).__init__()
        self.dim = dim

    def forward(self, X):
        return torch.unsqueeze(X, dim=self.dim)


class Conv1dAutoEncoder(nn.Module):

    def __init__(self, params):
        super(Conv1dAutoEncoder, self).__init__()
        self.params = params
        self.build()

    def build(self):
        encoder_mods = []
        for p in self.params:
            if p['type'] == 'conv':
                k = p.get('kernel', 3)
                pad = k // 2 if p.get('pad', 'same') != 'valid' else 0
                encoder_mods.append(
                    nn.Conv1d(
                        p['in'], p['out'], k, p.get('stride', 1), pad,
                        p.get('dilation', 1), p.get('groups', 1),
                        p.get('bias', False)
                    )
                )
                encoder_mods.append(nn.BatchNorm1d(p['out']))
                encoder_mods.append(p.get('activation', nn.ReLU)())
                if 'pool' in p:
                    encoder_mods.append(make_pool1d_layer(p['pool']))
            elif p['type'] == 'dense':
                encoder_mods.append(
                    nn.Linear(p['in'], p['out'], p.get('bias', False))
                )
                encoder_mods.append(p.get('activation', nn.ReLU)())
            elif p['type'] == 'squeeze':
                encoder_mods.append(Squeezer())

        # package encoding layers as a Sequential network
        self.encoder_net = nn.Sequential(*encoder_mods)

    def encode(self, X):
        return self.encoder_net(X)

    def decode(self, X):
        for layer in reversed(self.encoder_net):
            if isinstance(layer, nn.Conv1d):
                pad = layer.weight.shape[-1] // 2
                st_pad = layer.stride[0] // 2

                X = F.conv_transpose1d(
                    X, layer.weight, layer.bias, layer.stride, pad, st_pad,
                    layer.groups, layer.dilation
                )
            elif isinstance(layer, nn.Linear):
                X = F.linear(X, layer.weight.transpose(0, 1), layer.bias)
            elif isinstance(layer, Squeezer):
                X = torch.unsqueeze(X, dim=2)
            # elif not isinstance(layer, nn.BatchNorm1d):
            else:
                X = layer(X)
        return X

    def forward(self, X):
        # return self.encode(X)
        return self.decode(self.encode(X))


def ae_build_1():
    autoencoder = Conv1dAutoEncoder([
        {'type': 'conv', 'in': 1, 'out': 128, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 512, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 512, 'kernel': 5, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 256, 'kernel': 3, 'stride': 2},
        {
            'type': 'conv', 'in': 256, 'out': 128, 'kernel': 11, 'stride': 11,
            # 'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 2},
    ])
    return autoencoder


if __name__ == '__main__':
    autoencoder = ae_build_1()
    a = torch.randn(10, 1, 350)
    print(autoencoder.forward(a).shape)

    # issues with padding calculations and stride are causing mismatch of
    # downsampling -> transpose upsampling, so output is smaller than input.
