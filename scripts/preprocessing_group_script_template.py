"""Template script for running EEG processing / analysis across a group.

Moving from notebook -> script

1) Generalize, in the notebook, load file procedure
    - Have data all in one folder, in the notebook scan the folder, get a list of all files
    - Filter files: files = [file_name for file_name in subj_files if '.mat' in file_name]
"""

## IMPORTS
import os
import fnmatch
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import mne

###################################################################################################
###################################################################################################

# SETTINGS

#  Anything that is chosen variable, across all subjects
#    Anything that we might change
#    Purpose: everything in one place

# Note: Settings are defined as globals
#  This makes them accessible from within the 'main' function

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
    data_path = '/Users/jarrodmhicks/Desktop/Data/'
    subjects = [file for file in os.listdir(data_path) if fnmatch.fnmatch(file, 'A*')]

    # Loop across all subjects
    for subj_ind, subj_file in enumerate(subjects):

        # Add status updates
        print('Running Subject # ', subj_ind)

        # Load subject of data
        subj_path = os.path.join(data_path, subj, 'EEG', 'raw', 'raw_format')
        subj_files = os.listdir(subj_path)
        subj_files = [file for file in subj_files if fnmatch.fnmatch(file, '*.raw')]

        # ALSO ADD SOME FILE SELECTION SOMEWHERE HERE on subj_files list
        # INSERT CODE TO LOAD DATA

        # Do a pre-processing

        # Load standard montage, drop misc channels, and display channel locations# Load
        montage = mne.channels.read_montage('GSN-HydroCel-129', ch_names=raws[0].ch_names)

        for raw in raws:
            raw.set_montage(montage)
            raw.drop_channels(raw.ch_names[128:-1])

        # Do analyses of interest

        # Note: might have to add try/excepts for problems
        try:
            pass
        except:
            print('Subject number' subj_ind, 'failed.')

        # Note: remember to collect things of interest into group stores
        #  and/or save out individual files (however makes sense)

    # Save any group level files
    np.save(outputs, 'check')


if __name__ == "__main__":
    main()
