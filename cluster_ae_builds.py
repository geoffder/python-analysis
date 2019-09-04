from torch import nn
from conv1d_deep_cluster import Conv1dDeepClusterer


def ae_build_1():
    """Works with length 352"""
    autoencoder = Conv1dDeepClusterer([
        {'type': 'conv', 'in': 1, 'out': 128, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 512, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 512, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 256, 'kernel': 11, 'stride': 2},
        {
            'type': 'conv', 'in': 256, 'out': 128, 'kernel': 11, 'stride': 1,
            'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 10},
    ])
    return autoencoder


def ae_build_2():
    """Works with length 120"""
    autoencoder = Conv1dDeepClusterer([
        {
            'type': 'conv', 'in': 1, 'out': 128, 'kernel': 11, 'stride': 1,
            'activation': nn.Tanh
        },
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 5, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 256, 'kernel': 3, 'stride': 1},
        {'type': 'conv', 'in': 256, 'out': 256, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 256, 'kernel': 3, 'stride': 1},
        {'type': 'conv', 'in': 256, 'out': 512, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 256, 'kernel': 3, 'stride': 1},
        {
            'type': 'conv', 'in': 256, 'out': 128, 'kernel': 15, 'stride': 1,
            'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 10, 'activation': nn.Tanh},
    ])
    return autoencoder


def ae_build_3():
    """Works with length 352"""
    autoencoder = Conv1dDeepClusterer([
        {'type': 'conv', 'in': 1, 'out': 128, 'kernel': 11, 'stride': 1},
        {'type': 'conv', 'in': 128, 'out': 128, 'kernel': 5, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 5, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 256, 'kernel': 3, 'stride': 1},
        {'type': 'conv', 'in': 256, 'out': 512, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 512, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 512, 'out': 256, 'kernel': 3, 'stride': 2},
        {
            'type': 'conv', 'in': 256, 'out': 128, 'kernel': 11, 'stride': 1,
            'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 5},
    ])
    return autoencoder


def ae_build_4():
    """Works with length 383"""
    autoencoder = Conv1dDeepClusterer([
        {'type': 'conv', 'in': 1, 'out': 64, 'kernel': 11, 'stride': 4},
        {'type': 'conv', 'in': 64, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 64, 'kernel': 3, 'stride': 2},
        {'type': 'flatten'},
        {'type': 'dense', 'in': 384, 'out': 12},
    ])
    return autoencoder


def ae_build_5():
    """Works with length 383"""
    autoencoder = Conv1dDeepClusterer([
        {'type': 'conv', 'in': 1, 'out': 64, 'kernel': 11, 'stride': 4},
        {'type': 'conv', 'in': 64, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 64, 'kernel': 3, 'stride': 2},
        {
            'type': 'conv', 'in': 64, 'out': 128, 'kernel': 6, 'stride': 1,
            'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 12},
    ])
    return autoencoder


def ae_build_6():
    """Works with length 384"""
    autoencoder = Conv1dDeepClusterer([
        {
            'type': 'conv', 'in': 1, 'out': 64, 'kernel': 3, 'stride': 2,
            # 'pool': {'op': 'avg', 'kernel': 2}
        },
        {'type': 'conv', 'in': 64, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 64, 'kernel': 3, 'stride': 2},
        {
            'type': 'conv', 'in': 64, 'out': 128, 'kernel': 12, 'stride': 1,
            'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 12},
    ])
    return autoencoder


def ae_build_7():
    """Works with length 383"""
    autoencoder = Conv1dDeepClusterer([
        {'type': 'conv', 'in': 1, 'out': 32, 'kernel': 11, 'stride': 2},
        {'type': 'conv', 'in': 32, 'out': 64, 'kernel': 5, 'stride': 2},
        {'type': 'conv', 'in': 64, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 256, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 256, 'out': 128, 'kernel': 3, 'stride': 2},
        {'type': 'conv', 'in': 128, 'out': 64, 'kernel': 3, 'stride': 2},
        {
            'type': 'conv', 'in': 64, 'out': 128, 'kernel': 6, 'stride': 1,
            'pad': 'valid'
        },
        {'type': 'squeeze'},
        {'type': 'dense', 'in': 128, 'out': 12},
    ])
    return autoencoder
