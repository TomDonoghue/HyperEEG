""""Group analysis script to perform classification"""

# %matplotlib inline

import os
from copy import deepcopy
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from mne import read_epochs
from mne import set_log_level
set_log_levelset_log ('ERROR') # Keep MNE quiet

from scipy.stats import zscore, sem

from hypertools.tools.align import align

from sklearn import svm
from sklearn.model_selection import cross_val_score

###################################################################################################
###################################################################################################

# HELPER FUNCTIONS

def extract_data(dat):
    # INSERT TWO ARGUMENTS FOR FILTERING AND DOWNSAMPLING
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

    return np.stack(np.split(dat, n_epochs, 1))


def within_subj_classification(all_data, all_labels):
    """Run within subject classification within each subject for a list of subjects data.

    Parameters
    ----------
    data : list of 3d array
        xx
    labels : list of 1d array
        xx

    Returns
    -------
    scores : 1d array
        xx
    """

    # Run cross-validated classification within each subject
    scores = np.zeros(shape=[len(all_data), K_FOLD])
    for s_ind, subj_data, subj_labels in zip(range(n_subjs), all_data, all_labels):
        scores[s_ind, :] = cross_val_score(CLF, feature_dat(subj_data), subj_labels, cv=K_FOLD)

    return scores


def time_within_subj_classification(all_data, all_labels):
    """Run within subject classification stepping across each time point."""

    n_times = np.shape(all_data[0])[2]

    scores = np.zeros(shape=[len(all_data), n_times, K_FOLD])

    for s_ind, subj_data, subj_labels in zip(range(n_subjs), all_data, all_labels):
        for t_ind, t_step in enumerate(subj_data.T):
            scores[s_ind, t_ind, :] = cross_val_score(CLF, t_step.T, subj_labels, cv=K_FOLD)

    return scores

def time_btwn_subj_classification(all_data, all_labels):
    """   """

    n_times = np.shape(all_data[0])[2]
    scores = np.zeros(shape=[len(all_data), n_times])

    for s_ind, subj_data, subj_labels in zip(range(len(all_data)), all_data, all_labels):

        # Take a copy of the group data, and drop held out subject
        temp_data = deepcopy(all_data)
        temp_labels = deepcopy(all_labels)
        del temp_data[s_ind]
        del temp_labels[s_ind]

        group_data = np.concatenate(temp_data, 0)
        group_labels = np.concatenate(temp_labels, 0)

        for t_ind, t_step in enumerate(subj_data.T):

            # Train on group @ time point & classify left out subject @ time point
            CLF.fit(group_data[:, :, t_ind], group_labels)
            scores[s_ind, t_ind] = CLF.score(t_step.T, subj_labels)

    return scores


def btwn_subj_classication(all_data, all_labels):
    """Run classification between subjects.

    Parameters
    ----------
    all_data : list of 3d array
        Data for each subject.
    all_labels : list of 1d array
        Labels for each subject.

    Returns
    -------
    scores : list of float
        The classifications scores for each held out subject, as predicted from the group.
    """

    scores = [None] * len(all_data)

    for ind, subj_data, subj_labels in zip(range(len(all_data)), all_data, all_labels):

        # Take a copy of the group data, and drop held out subject
        temp_data = deepcopy(all_data)
        temp_labels = deepcopy(all_labels)
        del temp_data[ind]
        del temp_labels[ind]

        # Collapse group for training the model
        group_data = feature_dat(np.concatenate(temp_data, 0))
        group_labels = np.concatenate(temp_labels, 0)

        # Train on group & classify left out subject
        CLF.fit(group_data, group_labels)
        scores[ind] = CLF.score(feature_dat(subj_data), subj_labels)

    return scores


def maxabs(dat, dim):
    return np.max(np.abs(dat), dim)


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


def print_avg(label, score):
    print(label + ': {:1.2f}%'.format(score *100))


def print_avgs(label, scores):
    print(label + ':')
    for ind, score in enumerate(scores):
        print('\t{:1.0f} \t {:1.2f}'.format(ind, score))


def plot_results(times, results, labels):
    """Plot results of each decoding"""

    fig, ax = plt.subplots(figsize=(14, 6))

    for result in results:
        plot_time_class(times, np.mean(result, 0), sem(result), ax)

    # Chance line @ 50%
    plot_time_class(times, np.ones(len(times)) * 0.5, ax=ax, color='grey', alpha=0.7)

    # Hard set axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([0.4, 0.9])

    # Add axis labels
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14)

    ## Aesthetics

    #ax.grid(True)

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Add a legend
    plt.legend(labels, fontsize=14)


def plot_time_class(times, dat, err=None, ax=None, *args, **kwargs):
    """Plot  classification accuracy versus time"""

    if not ax:
        fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(times, dat, *args, **kwargs)

    # Standard error shading
    if np.any(err):
        ax.fill_between(times, dat-err, dat+err, alpha=0.5)

###################################################################################################
###################################################################################################

# SETTINGS

# Set the collection of ways to average across features
# Note that this is only used when NOT decoding across time
AVGS = {
    'maxabs' : maxabs,
    'max' : np.max,
    'min' : np.min,
    'mean' : np.mean,
    'median' : np.median
}

# Classification Settings
K_FOLD = 3
AVG_TO_USE = 'mean'

# Initialize SVM classification object
#CLF = svm.SVC(kernel='linear')
CLF = svm.LinearSVC()

# Set data size
# Note: these are set for arbitrary test data - UPDATE for 'real' data
n_epochs = 40
n_chs = 128
#n_times = 1001
n_per_cond = int(n_epochs / 2)

###################################################################################################
###################################################################################################

# LOAD & ORGANIZE DATA

# Set data location for processed files
dat_path = '/Users/tom/Desktop/HyperEEG_Project/Data/proc/'

# Get list of available files
#  Note: this currently excludes first subject, because they are weird.
dat_files = [file for file in os.listdir(dat_path)[1:] if 'epo.fif' in file]

# Load all data
all_subjs = [read_epochs(os.path.join(dat_path, f_name),
                         preload=True, verbose=None) for f_name in dat_files]

raise ValueError('Update the extract_data function for filtering and resampling')
# Downsample data (ANDY IS INCORPORATING INTO EXTRACT DATA FUNCTION)
# all_subjs = [dat.resample(100) for dat in all_subjs]

# Grab a time definition
times = all_subjs[0].times

# Check how many subjects there are
n_subjs = len(all_subjs)

# Organize subject data into data and label matrices
all_data, all_labels = [], []

# Finally extract the data we want from each subject
for subj in all_subjs:

    # Enforce a minimum number of trials - skip subj if not met
    if len(subj) < n_epochs:
        print('Skipping Subj')
        continue

    t_data, t_labels = extract_data(subj)
    all_data.append(t_data)
    all_labels.append(t_labels)

###################################################################################################
###################################################################################################

# HYPERALIGNMENT

# Data organization - extract matrices, and flatten to continuous data
all_data = [make_2d(dat) for dat in all_data]

# Do alignment
#  Note: this also switches orientation (takes the transpose) to match hypertools
aligned_data = align([dat.T for dat in all_data]) # Note: align assumes [n_samples x n_channels]
aligned_data = [dat.T for dat in aligned_data]
aligned_data = [make_3d(dat) for dat in aligned_data]

###################################################################################################
###################################################################################################

# CLASSIFICATION

# Run within subject classification - non time resolved
within_scores = within_subj_classification(all_data, all_labels)

# Get average results - within and across subjects
within_subj_avgs = np.mean(within_scores, 1)
within_glob_avg = np.mean(within_subj_avgs)

# # Check outcome - average across all subjects
# print_avg('CV Within-Subj Prediction', within_glob_avg)
#
# # Check performance on each subject
# print_avgs('Per Subj Within Predictions', within_subj_avgs)

# Within subject classification - time resolved
ts_within_scores = time_within_subj_classification(all_data, all_labels)

# Collapse across k-folds
ts_within_scores = np.mean(ts_within_scores, 2)


# Run prediction between subjects - on unaligned data
btwn_scores = btwn_subj_classication(all_data, all_labels)

# Get average results
avg_btwn_scores = np.mean(btwn_scores)

# # Check outcome - average across all subjects
# print_avg('Btwn-Subj Prediction', avg_btwn_scores)
#
# # Check performance on each subject (non-time resolved)
# print_avgs('Btwn Subject Classification', btwn_scores)

# Run prediction between subjects - on unaligned data and time-resolved
ts_btwn_scores = time_btwn_subj_classification(all_data, all_labels)


# Check within subject prediction of aligned data
within_al1_scores = within_subj_classification(aligned_data, all_labels)

# # Check outcome - average across all subjects
# print_avg('Within Aligned', np.mean(within_al1_scores))

# # Check performance on each subject (non-time resolved)
# print_avgs('\nPer subj Within-Aligned', np.mean(within_al1_scores, 1))

# Run prediction between subjects - on aligned data
btwn_al_scores = btwn_subj_classication(aligned_data, all_labels)

# Get average results
avg_btwn_al_scores = np.mean(btwn_al_scores)

# # Check outcome - average across all subjects
# print_avg('Btwn-Subj Prediction', avg_btwn_al_scores)
#
# # Check performance on each subject
# print_avgs('Btwn Subject Classification', btwn_al_scores)

# Run prediction between subjects - on aligned data, across time points
ts_btwn_al_scores = time_btwn_subj_classification(aligned_data, all_labels)

# Calculate average across time points
avg_time_class_btwn_al = np.mean(ts_btwn_al_scores, 0)

# Set up data for plotting
#results = [avg_time_class_within, avg_time_class_btwn, avg_time_class_btwn_al]
results = [ts_within_scores, ts_btwn_scores, ts_btwn_al_scores]
labels = ['Within', 'Between', 'Btwn-Al']

# Make an amazing plot
plot_results(times, results, labels)
