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

    minis = np.nan_to_num(np.concatenate([
        pd.read_csv(os.path.join(pth, fname), skiprows=2).values
        for fname in fnames
    ], axis=1))

    return minis.T


def clip_traces(minis, start, end):
    """
    Default mini output from TaroTools has 40ms head, and 60ms tail, centred
    around the peak. Often, recordings are 10kHz -> 10 samples/ms.
    Note: Slicing is done by pts, not temporal scaling.
    """
    return {k: v[:, start:end] for k, v in minis.items()}


def self_normalizer(minis):
    """
    Normalizes traces by their largest absolute value (so it works for inward
    and outward going currents). Expects numpy array of shape (N, T).
    """
    minis = {
        k: np.abs(v) / (np.abs(v).max(axis=1).reshape(-1, 1) + .00001)
        for k, v in minis.items()
    }
    return minis


def grouped_normalizer(minis):
    """
    Takes the absolute value of, then normalizes traces by the largest value
    across ALL recordings. Expects dict containing arrays of shape (N, T).
    """
    extremum = np.max(
        np.abs(np.concatenate([v for v in minis.values()]))
    )
    return {k: np.abs(v) / extremum for k, v in minis.items()}


def create_dataset(minis, balance=False):
    """
    Given a dict contaning arrays of processed minis, shape:(N, T), create a
    return a labelled dataset suitable for machine learning applications.
    """
    # balance the samples of each group in the dataset
    if balance:
        N = np.max([recs.shape[0] for recs in minis.values()])
        minis = {
            k: np.concatenate([v]*(N//v.shape[0])) if N//v.shape[0] > 1 else v
            for (k, v) in minis.items()
        }
    # combine all recs into single array, and add singleton channel dimension
    miniset = np.concatenate([recs for recs in minis.values()])
    miniset = miniset.reshape(miniset.shape[0], 1, -1)
    # integer label array
    labels = np.concatenate([
        [i]*recs.shape[0]
        for (i, recs) in enumerate(minis.values())
    ])
    # lookup for integer label -> actual label
    label_strs = [k for k in minis.keys()]

    return miniset, labels, label_strs


def get_minis_dataset(pth, start=370, end=490, norm='self'):
    """
    Load and process minis for autoencoder model.
    """
    minis = {
        trans: load_mini_csvs(pth, trans)
        for trans in ['ACh', 'GABA']
    }

    minis = clip_traces(minis, start, end)
    if norm == 'self':
        minis = self_normalizer(minis)
    else:
        minis = grouped_normalizer(minis)

    minis, labels, label_strs = create_dataset(minis)

    return minis, labels, label_strs


if __name__ == '__main__':
    datapath = "/media/geoff/Data/ss_minis/"

    minis = {
        trans: load_mini_csvs(datapath, trans)
        for trans in ['ACh', 'GABA', 'mixed']
    }

    minis = self_normalizer(clip_traces(minis, 350, 702))

    X, T, Tlabels = create_dataset(minis)
