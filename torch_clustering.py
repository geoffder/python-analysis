import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


def target_distribution(probs):
    """
    More certain cluster probabilities to serve as target for cluster loss.
    Takes soft cluster assignments and pushes them 'harder' (-> argmax).
    """
    weight = probs**2 / probs.sum(dim=0)
    return weight / weight.sum(dim=1).unsqueeze(1)


def forgy_init(X, K):
    """Returns random samples to initialize as K cluster centres."""
    return X[np.random.choice(X.shape[0], K)]


def calc_distances(X, centres):
    """
    Returns distances of each sample from each cluster centre.
    Shapes :: X:(N, D), centres:(K, D) -> distances:(N, K)
    """
    distances = torch.stack([
        (X - centre.unsqueeze(0)).pow(2).sum(dim=1)
        for centre in centres
    ], dim=1)
    return distances


def calc_dispersion(X, centres, cluster_probs):
    dists = (centres-X.mean(dim=0)).pow(2).sum(dim=1)
    return dists @ cluster_probs.sum(dim=0).unsqueeze(1)


def hard_cluster_index(X, clusters, centres):
    """Distance of samples from their assigned cluster centre."""
    error = torch.cat([
        X[clusters == i] - centre for i, centre in enumerate(centres)
    ]).pow(2).sum()
    return error / X.shape[0]


def soft_cluster_index(X, centres):
    """
    Clustering error estimated by sum of point -> cluster centre distances,
    weighted by the cluster assignment probabilities of each point.
    """
    dists = calc_distances(X, centres)
    probs = soft_assign_clusters(dists)
    return (dists * probs).sum() / X.shape[0]


def calinski(X, centres):
    K = centres.shape[0]
    dists = calc_distances(X, centres)
    probs = soft_assign_clusters(dists**2)  # experiment with squaring
    between = calc_dispersion(X, centres, probs)
    within = ((dists * probs).sum() / X.shape[0])
    return -(between * (X.shape[0] - K)) / (within * (K - 1))


def hard_adjust_centres(X, clusters, K):
    """
    Centres move to mean of current cluster assignments.
    Shapes :: X:(N, D), clusters:(N,) -> centres:(K, D)
    """
    centres = torch.stack([
        torch.mean(X[clusters == i], dim=0)
        for i in range(K)
    ], dim=0)
    return centres


def soft_adjust_centres(X, cluster_probs, K):
    """
    Move centres, weighted by cluster assignment probabilities.
    Shapes :: X:(N, D), cluster_probs:(N, K) -> centres:(K, D)
    """
    return (cluster_probs.t() @ X)/cluster_probs.sum(dim=0).unsqueeze(1)


def hard_assign_clusters(distances):
    """Points are assigned to nearest cluster centre."""
    return torch.argmin(distances, dim=1)


def soft_assign_clusters(distances):
    """Calculate probabilities that points belong to each cluster."""
    return F.softmin(distances, dim=1)


def soft_assign_alt(distances):
    numer = 1 / (1 + distances)
    return numer / numer.sum(dim=1, keepdim=True)


def hard_kmeans(X, K, min_delta=1e-3):
    """
    K-means pytorch implementation using hard argmax cluster assignment.
    """
    # initialize centres at random samples
    centres = forgy_init(X, K)

    delta = 10000  # ghetto do while
    while delta**2 > min_delta:
        # distances between points and centres and used to assign clusters
        dists = calc_distances(X, centres)
        clusters = hard_assign_clusters(dists)

        # update centres based on new assignments
        centres_old = centres.clone()
        centres = hard_adjust_centres(X, clusters, K)

        # calculate centre shift, halt when it drops below minimum
        delta = (centres - centres_old).pow(2).sum(dim=1).sqrt().sum()

    error = hard_cluster_index(X, clusters, centres)
    return centres, clusters, error


def soft_kmeans(X, K, min_delta=1e-6):
    """
    K-means pytorch implementation using softmax cluster probability
    assignment, rather than hard assigment with argmax. This way the algorithm
    is fully differentiable, allowing for gradient descent.
    """
    # initialize centres at random samples
    centres = forgy_init(X, K)

    delta = 10000  # ghetto do while
    while delta**2 > min_delta:
        # distances between points and centres and used to assign clusters
        dists = calc_distances(X, centres)
        cluster_probs = soft_assign_clusters(dists)

        # update centres based on new assignments
        centres_old = centres.clone()
        centres = soft_adjust_centres(X, cluster_probs, K)

        # calculate centre shift, halt when it drops below minimum
        delta = (centres - centres_old).pow(2).sum(dim=1).sqrt().sum()

    # error = soft_cluster_index(X, centres)
    error = calinski(X, centres)
    return centres, cluster_probs, error


class ClusterLayer(nn.Module):

    def __init__(self, K, D):
        super(ClusterLayer, self).__init__()
        self.K = K
        self.D = D
        self.dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()
        self.to(self.dv)

    def build(self):
        self.centres = nn.Parameter(
            torch.randn(self.K, self.D)/np.sqrt(self.D)
        )

    def forward(self, X):
        distances = torch.stack([
            (X - centre.unsqueeze(0)).pow(2).sum(dim=1)
            for centre in self.centres
        ], dim=1)
        return distances

    def fit(self, X, lr=1e-2, epochs=100, batch_sz=100, print_every=30,
            show_plot=False):

        N = X.shape[0]
        # X = torch.from_numpy(X).float().to(self.dv)
        X = X.to(self.dv)

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
        dists = self.forward(inputs)
        probs = soft_assign_clusters(dists)
        loss = F.kl_div(probs, target_distribution(probs).detach())

        # between = calc_dispersion(inputs, self.centres, probs)
        # within = ((dists * probs).sum() / inputs.shape[0])
        # loss = -(between/within) * ((inputs.shape[0]-self.K) / (self.K-1))
        # loss = -(between*(inputs.shape[0]-self.K))/(within*(self.K-1))
        # loss = within

        # Backward
        loss.backward()
        self.optimizer.step()  # Update parameters

        return loss.item()


if __name__ == '__main__':
    dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run k-means on toy data
    data = torch.cat([
        torch.randn(1000, 2) + i*10 for i in range(3)
    ], dim=0).to(dv)
    # centres, clusters, err = soft_kmeans(data, 3)

    data = (data - data.mean(dim=0)) / data.var(dim=0)
    clusterer = ClusterLayer(3, 2)
    clusterer.fit(data, epochs=50, lr=1e-5)
    centres = clusterer.centres
    print(centres)

    # back to numpy
    data = data.cpu().numpy()
    centres = centres.detach().cpu().numpy()
    # clusters = clusters.cpu().numpy()

    # plt.scatter(data[:, 0], data[:, 1], c=clusters, alpha=.1)
    plt.scatter(data[:, 0], data[:, 1], alpha=.1)
    plt.scatter(centres[:, 0], centres[:, 1], c='red', s=20)
    plt.show()
