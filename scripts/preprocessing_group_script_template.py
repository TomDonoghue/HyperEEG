"""Template script for running EEG processing / analysis across a group.

Moving from notebook -> script

1) Generalize, in the notebook, load file procedure
    - Have data all in one folder, in the notebook scan the folder, get a list of all files
    - Filter files: files = [file_name for file_name in subj_files if '.mat' in file_name]
"""

## IMPORTS
import os
from copy import deepcopy
import fnmatch
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.event import define_target_events
from mne.preprocessing import ICA

from faster import faster_bad_channels

from autoreject import AutoReject

###################################################################################################
###################################################################################################

# SETTINGS

#  Anything that is chosen variable, across all subjects
#    Anything that we might change
#    Purpose: everything in one place

# Note: Settings are defined as globals
#  This makes them accessible from within the 'main' function

# Settings
L_FREQ, H_FREQ = 0.1, 30. # filter settings
TMIN, TMAX = -1., 1. # epoch boundaries
BASELINE = (0.5, None) # period for baseline correction
EOG_CHS = ['E8', 'E14','E21','E25']

# Processing options
RUN_ICA = False
RUN_AUTOREJECT = False

# Set data path
DAT_PATH = 'path/to/data/files'

# Set other settings
MY_VAR = 'value'

###################################################################################################
###################################################################################################

# Running through the subjects
#  Everything from load & after - in a loop

def main():

    # SETUP
    # Any work that's outside the loop

    # Get all subj files (clean the list if needed)
    data_path = '/home/andrew/adykstra/Desktop/Data/'
    subjects = [file for file in os.listdir(data_path) if fnmatch.fnmatch(file, 'A*')]

    # Loop across all subjects
    for subj_ind, subj in enumerate(subjects):

        # Add status updates
        print('Running Subject # ', subj_ind)

        # Load subject of data
        subj_path = os.path.join(data_path, subj, 'EEG', 'raw', 'raw_format')
        subj_files = os.listdir(subj_path)
        subj_files = [file for file in subj_files if fnmatch.fnmatch(file, '*.raw')]

        # ALSO ADD SOME FILE SELECTION SOMEWHERE HERE on subj_files list
        # INSERT CODE TO LOAD DATA
        f_nums = ['06.raw', '08.raw', '10.raw'] # this needs to change
        raws = []
        for f_num in f_nums:
            try:
                raw_file = get_files(subj_files, f_num)
                raws.append(mne.io.read_raw_egi(os.path.join(subj_path, raw_file),
                                                preload=True, verbose=False))
            except:
                pass

        if len(raws) == 0:
            raise ValueError('There are no valid blocks for this subject.')
            
        # Set montage, drop misc channels, and filter
        montage = mne.channels.read_montage('GSN-HydroCel-129',
                                            ch_names=raws[0].ch_names)
        for raw in raws:
            raw.set_montage(montage)
            raw.drop_channels(raw.ch_names[128:-1])
            raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design='firwin',
                       verbose=False)
            
        # concatenate raw objects into single new raw object
        raw = mne.concatenate_raws(raws)
        
        # event handling
        events = mne.find_events(raw, verbose=False)
        trgts, rspns = [4, 5], [1, 2]

        # Create correct-response events
        sfreq = raw.info['sfreq']  # sampling rate
        correct_trials = [21, 22] # new event ids for correct responses.

        new_events = [] # list of new event objects
        for trgt, rspn, correct_trial in zip(trgts, rspns, correct_trials):
            tmp, _ = define_target_events(events, rspn, trgt, sfreq, tmin=-2., 
                                          tmax=0., correct_trial, fill_na=None)
            new_events.append(tmp)

        # collapse new events into an array
        new_events = np.concatenate((np.array(new_events[0]), np.array(new_events[1])),
                                    axis=0)
        new_event_ids = dict(left=21, right=22)
        
        # create epochs object
        epochs = mne.Epochs(raw, new_events, new_event_ids, tmin=TMIN, tmax=TMAX,
                            picks=None, baseline=BASELINE, reject=None,
                            preload=True, verbose=False)

        # mark bad channels via scrappy kurtosis z-score method        
        bad_channels = faster_bad_channels(epochs, thres=5)
        raw.info['bads'] = bad_channels
        epochs.info['bads'] = bad_channels        

        # re-reference to average reference
        raw.set_eeg_reference('average')
        epochs.set_eeg_reference('average')

        # high-pass filter for the purpose of ICA de-noising
        raw_hpf = deepcopy(raw)
        raw_hpf.filter(l_freq=1., h_freq=None, fir_design='firwin', verbose=False)
        
        epochs_hpf = mne.Epochs(raw_hpf, new_events, new_event_ids, tmin=TMIN,
                                tmax=TMAX, picks=None, baseline=BASELINE,
                                reject=None, preload=True, verbose=False)
        ica = ICA(random_state=1)
        ica.fit(epochs_hpf)
        
        # define bad components by correlating with channels near eyes
        eog_chs = [ch for ch in EOG_CHS if ch not in raw.info['bads']]
        
        bad_ica_comps = []
        for ch in eog_chs:
            inds, scores = ica.find_bads_eog(raw_hpf, ch_name=ch, threshold=3,
                                             l_freq=1, h_freq=8, verbose=False)
            bad_ica_comps.extend(inds)
            ica.plot_scores(scores, exclude=inds)

        bad_ica_comps = list(set(bad_ica_comps))
        ica.exclude = bad_ica_comps
        
        # plot and save bad components
        fig = ica.plot_components(picks=np.array(bad_ica_comps));
        fig_name = 'ica_scalp_maps.png'
        fig.savefig(fig_name, dpi=150)
        fig.close()
        
        # save ICA decomposition
        ica_filename = subj + '-ica.fif'
        ica.save(ica_filename)

        # apply ICA to both 
        epochs = ica.apply(epochs)
        raw = ica.apply(raw)

        raw_filename = subj + '-raw.fif'
        raw.save(raw_filename, overwrite=True)
        
        # use AutoReject to reject bad epochs and interpolate bad channels
        ar = AutoReject(random_state=1, verbose=False, cv=3)
        epochs, rej_log = ar.fit_transform(epochs, return_log=True)
        epochs.equalize_event_counts(new_event_ids, method='mintime')

        epochs_filename = subj + '_preprocessed-epo.fif'
        epochs.save(epochs_filename, overwrite=True)
        
        # Note: might have to add try/excepts for problems
        try:
            pass
        except:
            print('Subject number ' + str(subj_ind) + ' failed.')

        # Note: remember to collect things of interest into group stores
        #  and/or save out individual files (however makes sense)

    # Save any group level files
    np.save(outputs, 'check')


if __name__ == "__main__":
    main()
