"""Run through the MIPDB files and generate a list of task file mappings."""

import os
import csv

from mne import set_log_level
from mne.io import read_raw_egi

###################################################################################################
###################################################################################################

DAT_PATH = '/Users/tom/Desktop/HyperEEG_Project/Data/raw/'

# Create a mapping of event codes to file type
EVENTS = {
    'contrast'  : [{'12' : 1, '13' : 2, '5' : 3, '8' : 4, '9' : 5},
                   {'12' : 1, '13' : 2, '5' : 3, '8' : 4, '9' : 5, '94': 6},
                   {'12' : 1, '13' : 2, '5' : 3, '8' : 4, '9' : 5, '95': 6},
                   {'12' : 1, '13' : 2, '5' : 3, '8' : 4, '9' : 5, '96': 6}],

    'nat_view'  : [{'81': 33, '101': 1, '82': 34, '102': 2, '83': 35, '103': 3}],

    'resting'   : [{'20': 1, '30' : 2}, {'20': 1, '30' : 2, '90' : 3}],

    'seq_learn' : [{'11' :  1, '12':  2, '13' :  3, '14' :  4, '15' :  5, '16' :  6,
                    '17' :  7, '18':  8, '21' :  9, '22' : 10, '23' : 11, '24' : 12,
                    '25' : 13, '26': 14, '27' : 15, '28' : 16}],

    'surr_supp' : [{'4' : 1,  '8' : 2}],

    'symb_srch' : [{'14': 1, '20' : 2}],

    'the_nines' : [{'9999': 1}]
}

###################################################################################################
###################################################################################################

def cln_dict(evd):

    return {ke.strip(): va for ke, va in evd.items()}

def main():

    # Set MNE logging to only print critical issues
    set_log_level('CRITICAL')

    # Get list of all available subjects
    subj_nums = [file for file in os.listdir(DAT_PATH) if file[0] is not '.']

    file_labels = []
    for sub_num in subj_nums:

        subj_path = os.path.join(DAT_PATH, sub_num, 'EEG', 'raw', 'raw_format')
        subj_files = [file for file in os.listdir(subj_path) if ('A0' in file and file.endswith('.raw'))]

        print('\nRunning Subject #', sub_num)

        for ind, subj_file in enumerate(subj_files):

            raw = read_raw_egi(os.path.join(subj_path, subj_file), preload=False)

            cur_task = None

            # Special case: weird movie blocks
            if len(raw.event_id.keys()) > 25:
                    cur_task = 'movie'
                    print('\tSet event as misc / movie')
                    continue

            for key, vals in EVENTS.items():

                for val in vals:
                    if cln_dict(raw.event_id) == val:
                        print('\tFound event:', key)
                        cur_task = key
                        break

            if not cur_task:
                print('\tNo match found for file.')

            file_labels.append((subj_file[0:9], subj_file, cur_task))

    # Save out a file of all the mappings
    with open('file_mappings.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for f_info in file_labels:
            writer.writerow(list(f_info))


if __name__ == '__main__':
    main()
