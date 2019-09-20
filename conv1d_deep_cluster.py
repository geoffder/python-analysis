import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from mini_processing import get_minis_dataset, min_max_scaling
from temporal_convolution_1d import CausalConv1d
import torch_clustering as clorch
import cluster_ae_builds as builds


def make_pool1d_layer(param_dict):
    params = (
        param_dict['kernel'], param_dict.get('stride', param_dict['kernel']),
        param_dict.get('padding', 0)
    )
    if param_dict.get('op', 'avg') == 'max':
        return nn.MaxPool1d(*params)
    else:
        return nn.AvgPool1d(*params)


class Flatten(torch.nn.Module):
    """Flattens multi-dimensional input to create an shape:(N, D) matrix"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


class Squeezer(nn.Module):
    def __init__(self):
        super(Squeezer, self).__init__()

    @staticmethod
    def forward(X):
        return torch.squeeze(X)


class ClusterLossKm(nn.Module):

    def __init__(self, K, D, alpha=1e-3):
        super(ClusterLossKm, self).__init__()
        self.K = K
        self.D = D
        self.alpha = alpha

    def forward(self, X, encoding, decoding):
        decoder_loss = F.mse_loss(X, decoding)
        if self.alpha > 0:
            _, _, cluster_loss = clorch.soft_kmeans(encoding, self.K)
            # _, probs, _ = clorch.soft_kmeans(encoding, self.K)
            # cluster_loss = F.kl_div(
            #     probs.log(), clorch.target_distribution(probs).detach(),
            #     reduction='batchmean'
            # )
        else:
            cluster_loss = 0
        return decoder_loss + (self.alpha * cluster_loss / X.shape[0])


class ClusterLossCal(nn.Module):
    def __init__(self, K, D, alpha=1e-3):
        super(ClusterLossCal, self).__init__()
        self.K = K
        self.D = D
        self.alpha = alpha
        self.centres = nn.Parameter(
            torch.randn(self.K, self.D)/np.sqrt(self.D)
        )

    def forward(self, X, encoding, decoding):
        decoder_loss = F.mse_loss(X, decoding)
        if self.alpha > 0:
            cluster_loss = clorch.calinski(encoding, self.centres) / X.shape[0]
        else:
            cluster_loss = 0
        return decoder_loss + cluster_loss * self.alpha


class ClusterLossKLdiv(nn.Module):
    def __init__(self, K, D, alpha=1000):
        super(ClusterLossKLdiv, self).__init__()
        self.K = K
        self.D = D
        self.alpha = alpha
        # self.centres = nn.Parameter(
        #     torch.randn(self.K, self.D)/np.sqrt(self.D)
        # )
        self.centres = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.K, self.D))
        )

    def forward(self, X, encoding, decoding):
        decoder_loss = F.mse_loss(X, decoding)
        if self.alpha > 0:
            dists = clorch.calc_distances(encoding, self.centres)
            probs = clorch.soft_assign_clusters(dists)
            cluster_loss = F.kl_div(
                probs.log(), clorch.target_distribution(probs).detach(),
                reduction='batchmean'
            )
        else:
            cluster_loss = 0
        return decoder_loss + cluster_loss * self.alpha


class Conv1dDeepClusterer(nn.Module):

    def __init__(self, params):
        super(Conv1dDeepClusterer, self).__init__()
        self.params = params
        self.dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()
        self.to(self.dv)

    def build(self):
        encoder_mods = []
        for p in self.params:
            if p['type'] == 'conv':
                k = p.get('kernel', 3)
                dil = p.get('dilation', 1)
                pad = (k-1)*dil if p.get('pad', 'same') != 'valid' else 0

                if p.get('causal', False):
                    pad = pad // 2 if p.get('causal', False) else pad
                    encoder_mods.append(
                        nn.Conv1d(
                            p['in'], p['out'], k, p.get('stride', 1), pad, dil,
                            p.get('groups', 1), p.get('bias', False)
                        )
                    )
                else:
                    encoder_mods.append(
                        CausalConv1d(
                            p['in'], p['out'], k, p.get('stride', 1), pad, dil,
                            p.get('groups', 1), p.get('bias', False)
                        )
                    )

                encoder_mods.append(p.get('activation', nn.Tanh)())
                if 'pool' in p:
                    encoder_mods.append(make_pool1d_layer(p['pool']))
            elif p['type'] == 'pool':
                encoder_mods.append(make_pool1d_layer(p))
            elif p['type'] == 'dense':
                encoder_mods.append(
                    nn.Linear(p['in'], p['out'], p.get('bias', False))
                )
                encoder_mods.append(p.get('activation', nn.Tanh)())
            elif p['type'] == 'squeeze':
                encoder_mods.append(Squeezer())
            elif p['type'] == 'flatten':
                encoder_mods.append(Flatten())

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
                if layer.bias is not None:
                    X = X + layer.bias.unsqueeze(1)

                st_pad = layer.stride[0] // 2
                X = F.conv_transpose1d(
                    X, layer.weight, None, layer.stride, layer.padding,
                    st_pad, layer.groups, layer.dilation
                )
            elif isinstance(layer, CausalConv1d):
                if layer.conv.bias is not None:
                    X = X + layer.bias.unsqueeze(1)

                # No symmetrical padding in temporal dimension
                padding = 0
                st_pad = layer.conv.stride[0] // 2
                X = F.conv_transpose1d(
                    X, layer.conv.weight, None, layer.conv.stride, padding,
                    st_pad, layer.conv.groups, layer.conv.dilation
                )

                # remove all implicit T padding from the end (therefore causal)
                X = layer.chomp(X)

            elif isinstance(layer, nn.Linear):
                X = F.linear(X, layer.weight.transpose(0, 1), layer.bias)
            elif isinstance(layer, nn.AvgPool1d):
                X = F.interpolate(X, scale_factor=layer.stride)
            elif isinstance(layer, Squeezer):
                X = torch.unsqueeze(X, dim=2)
            elif isinstance(layer, Flatten):
                # dirty hack to determine the correct C dimensionality
                # [i-2] if one dense, [i-4] if two dense
                X = X.view(
                    X.shape[0],
                    list(self.encoder_net.children())[i-2].weight.shape[0],
                    -1
                )
            # elif not isinstance(layer, nn.BatchNorm1d):
            else:
                X = layer(X)
        return torch.tanh(X)

    def forward(self, X):
        Z = self.encode(X)
        return Z, self.decode(Z)

    def fit(self, X, K, lr=1e-4, epochs=10, batch_sz=300, cluster_alpha=1e-8,
            clust_mode='Km', print_every=30, show_plot=True):

        N = X.shape[0]
        D = list(self.encoder_net.children())[-2].weight.shape[0]
        X = torch.from_numpy(X).float().to(self.dv)

        if clust_mode == 'Km':
            self.loss = ClusterLossKm(K, D, alpha=cluster_alpha).to(self.dv)
        elif clust_mode == 'Cal':
            self.loss = ClusterLossCal(K, D, alpha=cluster_alpha).to(self.dv)
        elif clust_mode == 'KLdiv':
            self.loss = ClusterLossKLdiv(K, D, alpha=cluster_alpha).to(self.dv)

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
        encoding, decoding = self.forward(inputs)
        output = self.loss.forward(inputs, encoding, decoding)

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
            encoding, decoding = self.forward(inputs)
            output = self.loss.forward(inputs, encoding, decoding)

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
                _, constructs = self.forward(tensors)
                constructs = constructs.cpu().numpy()
                del tensors

            n_cols = batch_sz // 5
            fig, axes = plt.subplots(batch_sz // n_cols, n_cols)
            for ax, sample, construct in zip(
                    axes.flatten(), samples, constructs):
                ax.plot(np.squeeze(sample), label='sample')
                ax.plot(np.squeeze(construct), label='construct')
            plt.legend()
            plt.show()

            # again = input(
            #     "Show another reconstruction? Enter 'n' to quit\n"
            # )
            again = 'n'
            if again == 'n':
                break


if __name__ == '__main__':
    # change colours used for sequential plotting
    mpl.style.use('seaborn')

    datapath = "/media/geoff/Data/ss_minis/"

    print("Loading data...")
    length = 383
    minis, labels, label_strs = get_minis_dataset(
        datapath, start=350, end=350+length, norm='group_mean'
        # datapath, start=350, end=350+length, norm='feature'
    )

    print("Building network...")
    autoencoder = builds.ae_build_9()

    print("Fitting model...")
    autoencoder.fit(
        minis, 3, lr=1e-3, epochs=75, cluster_alpha=.01, clust_mode='KLdiv',
        # minis, 3, lr=1e-3, epochs=75, cluster_alpha=.25, clust_mode='KLdiv',
        # minis, 3, lr=1e-3, epochs=75, cluster_alpha=1.2, clust_mode='KLdiv',
        # minis, 4, lr=1e-3, epochs=75, cluster_alpha=.8, clust_mode='KLdiv',
        # minis, 2, lr=1e-4, epochs=75, cluster_alpha=.1, clust_mode='KLdiv',
        # minis, 3, lr=1e-3, epochs=75, cluster_alpha=2, clust_mode='Km',
        show_plot=False
    )

    print("Viewing dimensionality reduction...")
    reduced = autoencoder.get_reduced(minis)

    # Cluster reduced data, and calculate how labels are divided between the
    # obtained clusters.
    centres, clusters, _ = clorch.hard_kmeans(torch.from_numpy(reduced), 3)
    centres, clusters = centres.cpu().numpy(), clusters.cpu().numpy()
    counts, ratios = clorch.cluster_counts(clusters, labels)

    print("Cluster Breakdown:\n    "+(' '*5).join(label_strs)+'\n', ratios)
    if reduced.shape[1] > 2:
        # also, reduce the cluster centres (TSNE must do all at once)
        reduced_centres = TSNE(
            n_components=2, perplexity=75, learning_rate=400, n_iter=1000
        ).fit_transform(np.concatenate([reduced, centres], axis=0))
        # split samples and centres
        reduced = reduced_centres[:-centres.shape[0], :]
        centres = reduced_centres[-centres.shape[0]:, :]
        del reduced_centres

    fig, ax = plt.subplots(1, 2)

    # plot samples in 2d coordinate space, coloured by their true label
    for label in np.unique(labels):
        grp = reduced[labels == label]
        ax[0].scatter(grp[:, 0], grp[:, 1], label=label_strs[label], alpha=.5)

    # plot same samples, but coloured by their assigned cluster
    ax[1].scatter(reduced[:, 0], reduced[:, 1], c=clusters, alpha=.5)

    # plot cluster centres, with annotations stating what percentage of the
    # total population of each label resides there
    for (cx, cy), clstpop in zip(centres, ratios*100):
        ax[1].scatter(cx, cy, c='red')
        note = '\n'.join([
            "%d%% of %s" % (pop, lbl) for lbl, pop in zip(label_strs, clstpop)
        ])
        ax[1].annotate(
            note, (cx, cy), (10, 0), textcoords='offset pixels', c='red',
            fontsize=13, weight='heavy'
        )

    ax[0].legend()
    plt.show()

    # calculate average of minis grouped by their assigned clusters
    proto_events = np.concatenate([
        np.mean(minis[clusters == i], axis=0)
        for i in np.unique(clusters)
    ], axis=0)

    # normalize to mean=0, var=1
    proto_norms = (
        proto_events - proto_events.mean(axis=1, keepdims=True)
    ) / proto_events.std(axis=1, keepdims=True)

    # normalize on a 0 -> 1 scale (feature scaling)
    proto_stretch = min_max_scaling(proto_events)

    fig1, ax1 = plt.subplots(2, 1)
    ax1[0].plot(proto_events.T)
    ax1[1].plot(proto_stretch.T)
    plt.show()

    print("Viewing reconstructions...")
    autoencoder.reconstruct(minis, batch_sz=10)
