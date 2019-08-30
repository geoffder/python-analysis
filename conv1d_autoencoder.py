import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from mini_processing import get_minis_dataset


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


class Padder(nn.Module):
    def __init__(self):
        super(Padder, self).__init__()

    def forward(self, X):
        return F.pad(X, (0, 1)) if X.shape[-1] % 2 == 1 else X


class Conv1dAutoEncoder(nn.Module):

    def __init__(self, params):
        super(Conv1dAutoEncoder, self).__init__()
        self.params = params
        self.dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()
        self.to(self.dv)

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
                # encoder_mods.append(nn.BatchNorm1d(p['out']))
                encoder_mods.append(p.get('activation', nn.Tanh)())
                if 'pool' in p:
                    encoder_mods.append(make_pool1d_layer(p['pool']))
            elif p['type'] == 'dense':
                encoder_mods.append(
                    nn.Linear(p['in'], p['out'], p.get('bias', False))
                )
                encoder_mods.append(p.get('activation', nn.Tanh)())
            elif p['type'] == 'squeeze':
                encoder_mods.append(Squeezer())

        # package encoding layers as a Sequential network
        self.encoder_net = nn.Sequential(*encoder_mods)

    def encode(self, X):
        return self.encoder_net(X)

    def decode(self, X):
        for i, layer in enumerate(reversed(self.encoder_net)):
            # skip the last layer of the encoder, which is a non-linearity.
            # Do not want non-linearity -> non-linearity.
            if not i:
                continue

            if isinstance(layer, nn.Conv1d):
                st_pad = layer.stride[0] // 2
                X = F.conv_transpose1d(
                    X, layer.weight, layer.bias, layer.stride, layer.padding,
                    st_pad, layer.groups, layer.dilation
                )
            elif isinstance(layer, nn.Linear):
                X = F.linear(X, layer.weight.transpose(0, 1), layer.bias)
            elif isinstance(layer, Squeezer):
                X = torch.unsqueeze(X, dim=2)
            # elif not isinstance(layer, nn.BatchNorm1d):
            else:
                X = layer(X)
        return torch.tanh(X)

    def forward(self, X):
        return self.decode(self.encode(X))

    def fit(self, X, lr=1e-4, epochs=10, batch_sz=100, print_every=30,
            show_plot=True):

        N = X.shape[0]
        X = torch.from_numpy(X).float().to(self.dv)

        self.loss = nn.MSELoss().to(self.dv)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            print("epoch:", i, "n_batches:", n_batches)
            X = X[torch.randperm(X.shape[0])]  # shuffle
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                cost = self.train_step(Xbatch)  # train

                if j % print_every == 0:
                    print("cost: %f" % (cost))
                    costs.append(cost)

        if show_plot:
            plt.plot(costs)
            plt.show()

    def train_step(self, inputs):
        self.train()  # set the model to training mode
        self.optimizer.zero_grad()  # Reset gradient

        # Forward
        reconstruction = self.forward(inputs)
        output = self.loss.forward(reconstruction, inputs)

        # Backward
        output.backward()
        self.optimizer.step()  # Update parameters

        return output.item()

    def get_cost(self, inputs):
        """
        Get reconstruction loss without backprop step.
        """
        self.eval()  # set the model to testing mode
        self.optimizer.zero_grad()  # Reset gradient

        with torch.no_grad():
            # Forward
            reconstruction = self.forward(inputs)
            output = self.loss.forward(reconstruction, inputs)

        return output.item()

    def get_reduced(self, X):
        """
        Return reduced dimensionality hidden representation of X.
        """
        self.eval()  # set model to testing mode
        X = torch.from_numpy(X).float().to(self.dv)
        with torch.no_grad():
            reduced = self.encode(X).cpu().numpy()
        return reduced

    def reconstruct(self, X, batch_sz=5):
        self.eval()  # set the model to testing mode
        for i in range(50):
            inds = np.random.randint(0, X.shape[0], batch_sz)
            samples = X[inds]
            tensors = torch.from_numpy(samples).float().to(self.dv)

            with torch.no_grad():
                constructs = self.forward(tensors).cpu().numpy()
                del tensors

            fig, axes = plt.subplots(batch_sz, 1)
            for ax, sample, construct in zip(axes, samples, constructs):
                ax.plot(np.squeeze(sample))
                ax.plot(np.squeeze(construct))
            plt.show()

            again = input(
                "Show another reconstruction? Enter 'n' to quit\n"
            )
            # again = 'n'
            if again == 'n':
                break


def ae_build_1():
    """Works with length 352"""
    autoencoder = Conv1dAutoEncoder([
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
    autoencoder = Conv1dAutoEncoder([
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
    autoencoder = Conv1dAutoEncoder([
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


if __name__ == '__main__':
    datapath = "/media/geoff/Data/ss_minis/"

    print("Loading data...")
    minis, labels, label_strs = get_minis_dataset(
        # datapath, start=370, end=490, norm='self_max'
        datapath, start=350, end=702, norm='group_mean'
    )

    print("Building network...")
    autoencoder = ae_build_3()

    print("Fitting model...")
    autoencoder.fit(minis, lr=1e-4, batch_sz=300, epochs=30, show_plot=False)

    print("Viewing dimensionality reduction...")
    reduced = autoencoder.get_reduced(minis)

    if reduced.shape[1] > 2:
        reduced = TSNE(
            n_components=2, perplexity=30, learning_rate=100, n_iter=2000
        ).fit_transform(reduced)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, alpha=.3)
    plt.show()

    print("Viewing reconstructions...")
    autoencoder.reconstruct(minis)
