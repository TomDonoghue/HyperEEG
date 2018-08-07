""""Group analysis script to perform classification"""

import os

import numpy as np

from mne import read_epochs
from mne import set_log_level
set_log_level('ERROR') # Keep MNE quiet

from hypertools.tools.align import align

# Import custom code
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))

from hypeeg.analysis import (within_subj_classification, time_within_subj_classification,
                             btwn_subj_classication, time_btwn_subj_classification)
from hypeeg.utils import extract_data, make_2d, make_3d, print_avg, print_avgs
from hypeeg.plts import plot_results

###################################################################################################
###################################################################################################

# SETTINGS
from hypeeg.settings import *

# Set data location for processed files
DAT_PATH = '/Users/tom/Desktop/HyperEEG_Project/Data/proc/'

FILTER_SETTINGS = [('None', None, None), ('SCP', None, 5),
                   ('5-15', 5, 15), ('15_30', 15, 30)]

###################################################################################################
###################################################################################################

def main():

    print('\nRUNNNING ANALYSIS \n')
    print('\tRunning Alignment: ', ALIGN)

    # Get list of available files
    #  Note: this currently excludes first subject, because they are weird.
    dat_files = [file for file in os.listdir(DAT_PATH)[1:] if 'epo.fif' in file]

    # Grab a time definition
    times = np.arange(-1, 1, 1/100)

    ###################################################################################################

    for filter_setting in FILTER_SETTINGS:

        # Load (or reload) all subject data
        all_subjs = [read_epochs(os.path.join(DAT_PATH, f_name),
                             preload=True, verbose=None) for f_name in dat_files]

        f_label, f_low, f_high = filter_setting
        print('\t\tRunning filters:', f_label)

        all_data, all_labels = [], []

        for subj in all_subjs:

            # Enforce a minimum number of trials - skip subj if not met
            if len(subj) < N_EPOCHS:
                continue

            t_data, t_labels = extract_data(subj, l_freq=f_low, h_freq=f_high, resample=True)
            all_data.append(t_data)
            all_labels.append(t_labels)

        ###################################################################################################

        ## HYPERALIGNMENT

        # Data organization - extract matrices, and flatten to continuous data
        all_data_2d = [make_2d(dat) for dat in all_data]

        # Do alignment
        #   Note: this also switches orientation (takes the transpose) to match hypertools
        #     This is beacause align assumes data in orientation of [n_samples x n_channels]
        aligned_data = align([dat.T for dat in all_data_2d], align=ALIGN)
        aligned_data = [dat.T for dat in aligned_data]
        aligned_data = [make_3d(dat) for dat in aligned_data]

        ###################################################################################################

        ## CLASSIFICATION

        # Run within subject classification - non time resolved
        within_scores = within_subj_classification(all_data, all_labels)

        # Get average results - within and across subjects
        within_subj_avgs = np.mean(within_scores, 1)
        within_glob_avg = np.mean(within_subj_avgs)

        # Within subject classification - time resolved
        ts_within_scores = time_within_subj_classification(all_data, all_labels)

        # Collapse across k-folds
        ts_within_scores = np.mean(ts_within_scores, 2)

        # Run prediction between subjects - on unaligned data
        btwn_scores = btwn_subj_classication(all_data, all_labels)

        # Get average results
        avg_btwn_scores = np.mean(btwn_scores)

        # Run prediction between subjects - on unaligned data and time-resolved
        ts_btwn_scores = time_btwn_subj_classification(all_data, all_labels)

        # Check within subject prediction of aligned data
        within_al1_scores = within_subj_classification(aligned_data, all_labels)

        # Run prediction between subjects - on aligned data
        btwn_al_scores = btwn_subj_classication(aligned_data, all_labels)

        # Get average results
        avg_btwn_al_scores = np.mean(btwn_al_scores)

        # Run prediction between subjects - on aligned data, across time points
        ts_btwn_al_scores = time_btwn_subj_classification(aligned_data, all_labels)

        # Calculate average across time points
        avg_time_class_btwn_al = np.mean(ts_btwn_al_scores, 0)

        # Set up data for plotting
        results = [ts_within_scores, ts_btwn_scores, ts_btwn_al_scores]
        labels = ['Within', 'Between', 'Aligned']

        # Make an amazing plot
        plot_results(times, results, labels, save_fig=True, save_name=f_label + '_' + ALIGN)

if __name__ == '__main__':
    main()
