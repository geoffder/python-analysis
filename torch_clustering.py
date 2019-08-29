import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


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
    # return F.softmax((distances-distances.mean())/distances.var(), dim=1)
    return F.softmin(distances, dim=1)


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

    error = soft_cluster_index(X, centres)
    return centres, cluster_probs, error


if __name__ == '__main__':
    dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run k-means on toy data
    data = torch.cat([
        torch.randn(1000, 2) + i*10 for i in range(3)
    ], dim=0).to(dv)
    centres, clusters, err = soft_kmeans(data, 3)

    # back to numpy
    data = data.cpu().numpy()
    centres = centres.cpu().numpy()
    clusters = clusters.cpu().numpy()

    plt.scatter(data[:, 0], data[:, 1], c=clusters, alpha=.1)
    plt.scatter(centres[:, 0], centres[:, 1], c='red', s=20)
    plt.show()
