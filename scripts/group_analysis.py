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

from analysis import (within_subj_classification, time_within_subj_classification, ...
                      btwn_subj_classication, time_btwn_subj_classification, maxabs)

from utils import extract_data, make_2d, make_3d, print_avg, print_avgs

from plts import plot_results

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

    t_data, t_labels = extract_data(subj, l_freq=None, h_freq=5., resample=True)
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
