"""Utility functions for the HyperEEG project."""

from collections import Counter

import numpy as np
from scipy.stats import zscore

from settings import N_EPOCHS, AVGS, DEFAULT_AVG

###################################################################################################
###################################################################################################

def extract_data(dat, l_freq=None, h_freq=None, resample=False):
    """Organize data from MNE object, to data matrices and labels to be used for classification.

    Parameters
    ----------
    dat : mne.Epochs object
        A subject's worth of epoched data.

    Returns
    -------
    labels : 1d array
        Labels for each trial type.
    data : 3d array
        Epoched data matrix.
    """

    # Check event codes there are, and unpack
    ev_counts = Counter(dat.events[:, 2])
    evc_a, evc_b = dat.event_id.keys()
    n_evc_a, n_evc_b = ev_counts.values()

    # Generate labels
    lab_a = np.ones(shape=[n_per_cond]) * -1
    lab_b = np.ones(shape=[n_per_cond])

    # Filter data
    if l_freq is not None:
        dat.filter(l_freq=l_freq)
    if h_freq is not None:
        dat.filter(h_freq=h_freq)

    # Resample
    if resample:
        dat.resample(100)

    # Extract trial data
    eps_a = dat[evc_a]._data[0:n_per_cond, 0:128, :]
    eps_b = dat[evc_b]._data[0:n_per_cond, 0:128, :]

    # Check all our shapes and sizes are correct
    assert len(lab_a) == np.shape(eps_a)[0]
    assert len(lab_b) == np.shape(eps_b)[0]
    assert len(lab_a) == len(lab_b)
    assert np.shape(eps_a)[0] == np.shape(eps_b)[0]

    # Collect all labels and trial data together
    data = np.concatenate([eps_a, eps_b], 0)
    labels = np.hstack([lab_a, lab_b])

    return data, labels


def feature_dat(dat, avg_type=AVG_TO_USE):
    """Convert epochs

    Parameters
    ----------
    dat : 3d array
        xx
    avg_type : {'max', 'min', 'mean', 'median', 'maxabs'}
        xx

    Returns
    -------
    out : XX
        xx
    """

    avg = AVGS[avg_type]

    # Note: can add something here to select channels / time points
    out = avg(dat[:, :, :], 2)

    return out


def make_2d(dat, z_score=True):
    """Reorganize a 3D matrix into a continuous 2D matrix.

    Parameters
    ----------
    dat : 3d
        Epoched data matrix, as [n_epochs, n_channels, n_times]

    Returns
    -------
    2d array
        Continuous data matrix of epochs concatendat in time, as [n_channels, n_times_tot]
            Note: where n_times_tot = n_times * n_epochs
    """

    dat = np.concatenate(dat, 1)

    if z_score:
        dat = zscore(dat, 0)

    return dat


def make_3d(dat):
    """Reorganize a 2D matrix into the 3D trial structure matrix.

    Parameters
    ----------
    dat : 2d array
        Continuous data matrix of epochs concatendat in time, as [n_channels, n_times_tot]
            Note: where n_times_tot = n_times * n_epochs

    Results
    -------
    3d array
        Epoched data matrix, as [n_epochs, n_channels, n_times]
    """

    return np.stack(np.split(dat, N_EPOCHS, 1))


def print_avg(label, score):
    """   """

    print(label + ': {:1.2f}%'.format(score *100))


def print_avgs(label, scores):
    """   """

    print(label + ':')
    for ind, score in enumerate(scores):
        print('\t{:1.0f} \t {:1.2f}'.format(ind, score))
