import os

import numpy as np
import pandas as pd


def load_mini_csvs(pth, prefix):
    """
    Loads all CSVs with the given prefix in the given folder `pth`. Traces are
    returned as one numpy array of shape (N, T).
    """
    fnames = [
        f for f in os.listdir(pth)
        if prefix in f and os.path.isfile(os.path.join(pth, f))
    ]

    minis = np.concatenate([
        pd.read_csv(os.path.join(pth, fname), skiprows=2).values
        for fname in fnames
    ], axis=1)

    return minis.T


def normalizer(minis):
    """
    Normalizes traces by their largest absolute value (so it works for inward
    and outward going currents). Expects numpy array of shape (N, T).
    """
    return minis / np.abs(minis).max(axis=1).reshape(-1, 1)


if __name__ == '__main__':
    datapath = "/media/geoff/Data/ss_minis/"

    ach_minis = normalizer(load_mini_csvs(datapath, 'ACh'))
    gaba_minis = normalizer(load_mini_csvs(datapath, 'GABA'))
    mixed_minis = normalizer(load_mini_csvs(datapath, 'mixed'))
