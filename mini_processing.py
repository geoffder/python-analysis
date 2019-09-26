import os

import numpy as np
import pandas as pd

from scipy.signal import detrend
from scipy.optimize import curve_fit


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


def self_max_normalizer(minis):
    """
    Normalizes traces by their largest absolute value (so it works for inward
    and outward going currents). Expects numpy array of shape (N, T).
    """
    minis = {
        k: np.abs(v) / (np.abs(v).max(axis=1).reshape(-1, 1) + .00001)
        for k, v in minis.items()
    }
    return minis


def self_feature_scaling(minis):
    minis = {
        k: min_max_scaling(np.abs(v))
        for k, v in minis.items()
    }
    return minis


def self_mean_normalizer(minis):
    """
    Normalizes traces by their mean absolute value (so it works for inward
    and outward going currents) and variane. Expects numpy array of
    shape (N, T).
    """
    abs_minis = {k: np.abs(v) for k, v in minis.items()}

    minis = {
        k: (v - v.mean(axis=1).reshape(-1, 1))
        / (v.var(axis=1).reshape(-1, 1) + .00001)
        for k, v in abs_minis.items()
    }
    return minis


def grouped_max_normalizer(minis):
    """
    Takes the absolute value of, then normalizes traces by the largest value
    across ALL recordings. Expects dict containing arrays of shape (N, T).
    """
    extremum = np.max(
        np.abs(np.concatenate([v for v in minis.values()]))
    )
    return {k: np.abs(v) / extremum for k, v in minis.items()}


def grouped_mean_normalizer(minis):
    """
    Normalizes based on mean and variance of ALL minis across all groups
    (absolute valued so that inward and outward events are treated the same.)
    """
    all_abs_minis = np.abs(np.concatenate([v for v in minis.values()]))
    mean, var = np.mean(all_abs_minis), np.var(all_abs_minis)
    return {k: (np.abs(v) - mean) / var for k, v in minis.items()}


def detrend_minis(minis):
    return {k: detrend(v, axis=1) for k, v in minis.items()}


def min_max_scaling(waves):
    """Normalize on a 0 -> 1 scale, expects shape (N, T) numpy array"""
    waves = waves - waves.min(axis=1, keepdims=True)
    return waves / (waves.max(axis=1, keepdims=True) + .00001)


def balance_groups(minis):
    """
    Takes a dict of shape (N, T) numpy arrays and replicates samples of the
    smaller groups to roughly adjust for an imbalanced dataset.
    """
    N = np.max([recs.shape[0] for recs in minis.values()])
    minis = {
        k: np.concatenate([v]*(N//v.shape[0])) if N//v.shape[0] > 1 else v
        for (k, v) in minis.items()
    }
    return minis


def create_dataset(minis, balance=False):
    """
    Given a dict contaning arrays of processed minis, shape:(N, T), create a
    return a labelled dataset suitable for machine learning applications.
    """
    # balance the samples of each group in the dataset
    minis = balance_groups(minis) if balance else minis

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


def get_minis_dataset(pth, start=370, end=490, norm='self_max'):
    """
    Load and process minis for autoencoder model.
    """
    minis = {
        trans: load_mini_csvs(pth, trans)
        # for trans in ['ACh', 'GABA']
        for trans in ['ACh', 'GABA', 'mixed']
        # for trans in ['GABA', 'mixed']
    }

    # remove linear trends in offset if they exist
    minis = detrend_minis(minis)

    minis = clip_traces(minis, start, end)

    if norm == 'self_max':
        minis = self_max_normalizer(minis)
    elif norm == 'self_mean':
        minis = self_mean_normalizer(minis)
    elif norm == 'group_max':
        minis = grouped_max_normalizer(minis)
    elif norm == 'group_mean':
        minis = grouped_mean_normalizer(minis)
    elif norm == 'feature':
        minis = self_feature_scaling(minis)

    minis, labels, label_strs = create_dataset(minis)

    return minis, labels, label_strs


def expfun(X, y0, tau, bias):
    """Calculate Y values of exponential given X and parameters."""
    return y0 * np.exp((-1/tau) * X) + bias


def get_exp_fits(mini, peak_time):
    """
    Split waves based on position of peak (in pts), and calculates
    exponential fit parameters for the rise and decay portions of the
    event. Returns parameters (y0, tau, bias) for the rise and decay in
    a flat list.
    """
    head_y = np.flip(mini[:peak_time])
    tail_y = mini[peak_time-1:]

    head_x = np.arange(head_y.shape[0])
    tail_x = np.arange(tail_y.shape[0])

    (h_y0, h_tau, h_b), _ = curve_fit(expfun, head_x, head_y)
    (t_y0, t_tau, t_b), _ = curve_fit(expfun, tail_x, tail_y)

    return [h_y0, h_tau, h_b, t_y0, t_tau, t_b]


def moving_average(wave, kernel_sz=3):
    """
    Simple moving average of 1d wave with specifiable kernel size. Takes
    and returns a 1d numpy array.
    """
    cumul = np.cumsum(wave, dtype=float)
    cumul[kernel_sz:] = cumul[kernel_sz:] - cumul[:-kernel_sz]
    return cumul[kernel_sz - 1:] / kernel_sz


def get_rise_decay(wave, peak_time, kernel_sz=3):
    """
    Get number of points it takes for the given event to rise and decay
    between 20% <-> 80% of the peak difference over baseline.
    """
    # smooth input waves, flip head -> rise is modelled as a decay
    head_y = moving_average(np.flip(wave[:peak_time]), kernel_sz)
    tail_y = moving_average(wave[peak_time-1:], kernel_sz)

    baseline = head_y[:200].mean()
    peak = head_y.max()

    # values to measure the rise/decay time between
    bottom_thr = (peak - baseline) * .2 + baseline
    top_thr = (peak - baseline) * .8 + baseline

    # get point before wave drops below top threshold
    head_top = np.maximum(np.where(head_y < top_thr)[0][0] - 1, 0)
    tail_top = np.maximum(
        np.where(tail_y < top_thr)[0][0] if tail_y.max() > top_thr else 0,
        0
    )

    # find number of points from the rise/decay "top" point til threshold
    rise = np.where(head_y[head_top:] < bottom_thr)[0][0]
    decay = np.where(tail_y[tail_top:] < bottom_thr)[0][0]

    return [rise, decay]


def norm_dimensions(matrix):
    """
    Expects table of metrics in the form of a shape (N, D) numpy array.
    Each dimension is normalized to have mean=0 and var=1 over the population.
    """
    sub_mean = matrix - matrix.mean(axis=0)
    return sub_mean / matrix.var(axis=0)


if __name__ == '__main__':
    datapath = "/media/geoff/Data/ss_minis/"

    minis = {
        trans: load_mini_csvs(datapath, trans)
        for trans in ['ACh', 'GABA', 'mixed']
    }

    minis = grouped_mean_normalizer(clip_traces(minis, 350, 702))

    X, T, Tlabels = create_dataset(minis)
