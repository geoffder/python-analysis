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
        k: v / np.abs(v).max(axis=1).reshape(-1, 1) for k, v in minis.items()
    }
    return minis


def grouped_normalizer(minis):
    """
    Takes the absolute value of, then normalizes traces by the largest value
    across ALL recordings. Expects dict containing arrays of shape (N, T).
    """
    extremum = np.max(
        np.nan_to_num(np.abs(np.concatenate([v for v in minis.values()])))
    )
    return {k: np.abs(v) / extremum for k, v in minis.items()}


if __name__ == '__main__':
    datapath = "/media/geoff/Data/ss_minis/"

    minis = {
        trans: load_mini_csvs(datapath, trans)
        for trans in ['ACh', 'GABA', 'mixed']
    }

    minis = grouped_normalizer(clip_traces(minis, 350, 700))
