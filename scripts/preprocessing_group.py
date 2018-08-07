"""Group pre-processing script for the EEG HyperAlignment project.

Notes
-----
- I hope this works.
"""

import os
import csv
import pickle
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne import set_log_level
from mne.event import define_target_events
from mne.preprocessing import ICA

from autoreject import AutoReject
from faster import faster_bad_channels

###################################################################################################
###################################################################################################

# SETTINGS

# Set which task to run
TASK = 'contrast'

# EVENT CODES
#  Note: all event codes represent [LeftTrials, RightTrials]
TRGT_EVCS = [4, 5]          # Target event codes, as in raw data
RSPN_EVCS = [1, 2]          # Response event codes, as in raw data
CORR_EVCS = [21, 22]        # New event ids to use for correct responses.

# FILTER
L_FREQ, H_FREQ = 0.1, 30. # filter settings

# EPOCH SETTINGS
TMIN, TMAX = -1., 1. # epoch boundaries
BASELINE = (0.5, None) # period for baseline correction
EOG_CHS = ['E8', 'E14','E21','E25']
N_EPOCHS = 40 # minimum number of epochs a subject needs for analysis

# Processing options
RUN_ICA = True
RUN_AUTOREJECT = True

# Set paths
BASE_PATH = '/Users/tom/Desktop/HyperEEG_Project/Data/'
DATA_PATH = os.path.join(BASE_PATH, 'raw')
PROC_PATH = os.path.join(BASE_PATH, 'proc')
ICA_PATH  = os.path.join(BASE_PATH, 'ica')
FIG_PATH  = os.path.join(BASE_PATH, 'figs')

###################################################################################################
###################################################################################################

# Load the file that maps all the subject files to task type
fms = pd.read_csv('file_mappings.csv', header=None, names=['SUBNUM', 'FILE', 'TASK'])

# Running through the subjects
def main():

    # Initialize to keep track of how many events each subject has
    n_kept_events = []

    # Set the MNE print out level
    set_log_level('ERROR')

    # Get list of available subjects
    subnums = [name for name in os.listdir(DATA_PATH) if name[0] is not '.']

    # Loop across all subjects
    for idx, subnum in enumerate(subnums):

        # Add status updates
        print('\nRunning Subject # ', subnum)

        ## DATA LOADING

        print('\tData Wrangling')

        # Get the files that match the desired task, load and concatenate data
        subj_path = os.path.join(DATA_PATH, subnum, 'EEG', 'raw', 'raw_format')
        subj_files = list(fms.loc[(fms["SUBNUM"] == subnum) & (fms["TASK"] == TASK)]["FILE"].values)
        raws = [mne.io.read_raw_egi(os.path.join(subj_path, raw_file), preload=True) for raw_file in subj_files]

        # Set montage, drop misc channels, and filter
        montage = mne.channels.read_montage('GSN-HydroCel-129', ch_names=raws[0].ch_names)
        for raw in raws:
            raw.set_montage(montage)
            raw.drop_channels(raw.ch_names[128:-1])
            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design='firwin')

        # Concatenate raw objects into single new raw object
        raw = mne.concatenate_raws(raws)

        ## EVENT MANAGEMENT

        print('\tEvent Management')

        events = mne.find_events(raw, verbose=False)

        # Create correct-response events
        new_events = []

        for trgt, rspn, corr in zip(TRGT_EVCS, RSPN_EVCS, CORR_EVCS):

            tmp, _ = define_target_events(events, rspn, trgt, raw.info['sfreq'],
                                          tmin=-2., tmax=0., new_id=corr, fill_na=None)
            new_events.append(tmp)

        # Collapse new events into an array
        new_events = np.concatenate((np.array(new_events[0]), np.array(new_events[1])), axis=0)
        new_event_ids = dict(left=21, right=22)

        # Check if subject has enough epochs to continue
        if not np.any(new_events):
            print('Subject has no correct trials...')
            continue
        elif new_events.shape[1] < N_EPOCHS:
            print('Subject has too few trials for analysis')
            continue

        # Create epochs object
        epochs = mne.Epochs(raw, new_events, new_event_ids, tmin=TMIN, tmax=TMAX,
                            picks=None, baseline=BASELINE, reject=None, preload=True)

        ## BAD CHANNELS & RE-REFERENCING

        print('\tBad Channels & Re-Referencing')

        # Mark bad channels via scrappy kurtosis z-score method
        bad_channels = faster_bad_channels(epochs, thres=5)
        raw.info['bads'] = bad_channels
        epochs.info['bads'] = bad_channels

        # Re-reference to average reference
        raw.set_eeg_reference('average', projection=False)
        epochs.set_eeg_reference('average', projection=False)

        ## ICA

        if RUN_ICA:

            print('\tRunning ICA')

            # High-pass filter for the purpose of ICA de-noising
            raw_hpf = deepcopy(raw)
            raw_hpf.filter(l_freq=1., h_freq=None, fir_design='firwin', verbose=False)
            epochs_hpf = mne.Epochs(raw_hpf, new_events, new_event_ids,
                                    tmin=TMIN, tmax=TMAX, picks=None,
                                    baseline=BASELINE, reject=None, preload=True)
            ica = ICA(random_state=1)
            ica.fit(epochs_hpf)

            # Define bad components by correlating with channels near eyes
            eog_chs = [ch for ch in EOG_CHS if ch not in raw.info['bads']]

            bad_ica_comps = []
            for ch in eog_chs:
                inds, scores = ica.find_bads_eog(raw_hpf, ch_name=ch,
                                                 threshold=4, l_freq=1, h_freq=8)
                bad_ica_comps.extend(inds)

            ica.exclude = list(set(bad_ica_comps))

            # Plot and save bad components
            if len(bad_ica_comps) > 0:
                fig = ica.plot_components(picks=np.array(bad_ica_comps), show=False)
                fig_name = os.path.join(FIG_PATH, subnum + '_ica_scalp_maps.png')
                fig.savefig(fig_name, dpi=150)

            # Save out ICA decomposition
            ica_filename = os.path.join(ICA_PATH, subnum + '-ica.fif')
            ica.save(ica_filename)

            # Apply ICA to both epoched and raw data
            epochs = ica.apply(epochs)
            raw = ica.apply(raw)

            raw_filename = os.path.join(PROC_PATH, subnum + '-raw.fif')
            raw.save(raw_filename, overwrite=True)

        ## Autoreject

        if RUN_AUTOREJECT:

            print('\tRunning AutoReject')

            # Use AutoReject to reject bad epochs and interpolate bad channels
            ar = AutoReject(n_jobs=4, random_state=1, verbose=False, cv=3)
            epochs, rej_log = ar.fit_transform(epochs, return_log=True)
            epochs.info['bads'] = [] # no need for bad channels after AutoReject

            # Save out autoreject log, as a pickled object
            pickle.dump(rej_log, open(os.path.join(ICA_PATH, subnum + "-ar.p"), "wb"))

        ## SAVE OUT DATA

        # Enforce consistencty in the number of events per condition
        epochs.equalize_event_counts(new_event_ids, method='mintime')

        # Don't save the subject's data if they don't have enough epochs
        if len(epochs) < N_EPOCHS:
            print('Subject has too few trials for analysis')
            continue

        # Save out pre-processed data
        epochs_filename = os.path.join(PROC_PATH, subnum + '_preprocessed-epo.fif')
        epochs.save(epochs_filename)

        print('\tData Saved - {:2d} kept events'.format(len(epochs)))

        n_kept_events.append((subnum, len(epochs)))

        print('\nGreat Success.\n')

    # Save out the log file of number of good events per subject
    with open('event_log.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for ev_info in n_kept_events:
            writer.writerow(list(ev_info))


if __name__ == "__main__":
    main()
