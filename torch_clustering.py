import numpy as np
import torch


def forgy_init(X, K):
    """Returns random samples to initialize as cluster centres."""
    return X[np.random.choice(X.shape[0], K)]


def calc_distance(X, centres):
    pass


def adjust_centres():
    pass


def assign_clusters():
    pass


def kmeans(X, K, tolerance=1e-4):
    """
    lloyd K-means
    inspiration from:
    - https://github.com/overshiki/kmeans_pytorch
    - add more here if I use them until I've finished an implementation
    """
    pass

