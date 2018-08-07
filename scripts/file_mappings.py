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
    'contrast'  : [{'12' : 1, '13': 2, '5': 3, '8': 4, '9': 5}],

    'nat_view'  : [{'81': 33, '101':1, '82':34, '102':2, '83':35, '103':3}],

    'resting'   : [{'20': 1, '30' : 2}, {'20': 1, '30' : 2, '90' : 3}],

    'seq_learn' : [{'11' :  1, '12':  2, '13' :  3, '14' :  4, '15' :  5, '16' :  6,
                    '17' :  7, '18':  8, '21' :  9, '22' : 10, '23' : 11, '24' : 12,
                    '25' : 13, '26': 14, '27' : 15, '28' : 16}],

    'surr_supp' : [{'4' : 1,  '8' : 2}],

    'symb_srch' : [{'14': 1, '20' : 2}],

    'movie'     : [{'101' :  1, '102' :  2, '103' :  3, '104' :  4, '106':  5, '11' :  6, '12':  7,
                     '13' :  8,  '14' :  9,  '15' : 10,  '16' : 11,  '17': 12, '18' : 13, '20': 14,
                     '21' : 15,  '22' : 16,  '23' : 17,  '24' : 18,  '25': 19, '26' : 20, '27': 21,
                     '28' : 22,  '30' : 23,  '31' : 24,  '32' : 25,  '33': 26, '34' : 27, '35': 28,
                      '4' : 29,   '5' : 30,  '50' : 31,   '8' : 32,  '81': 33, '82' : 34, '83': 35,
                     '84' : 36,  '86' : 37,   '9' : 38,  '90' : 39,  '91': 40, '92' : 41, '93': 42,
                     '94' : 43,  '95' : 44,  '96' : 45,  '97' : 46}]
}

###################################################################################################
###################################################################################################

def cln_dict(evd):

    return {ke.strip(): va for ke, va in evd.items()}

def main():

    # Set MNE logging to only print critical issues
    set_log_level('CRITICAL')

    # Get list of all available subjects
    subj_nums = [file for file in os.listdir(dat_path) if file[0] is not '.']

    file_labels = []
    for sub_num in subj_nums:

        subj_path = os.path.join(dat_path, sub_num, 'EEG', 'raw', 'raw_format')
        subj_files = [file for file in os.listdir(subj_path) if ('A0' in file and file.endswith('.raw'))]

        print('Running Subject #', subj_file.split('.')[0])

        for ind, subj_file in enumerate(subj_files):

            raw = mne.io.read_raw_egi(os.path.join(subj_path, subj_file), preload=False)

            cur_task = None

            for key, vals in events.items():

                for val in vals:
                    if cln_dict(raw.event_id) == val:
                        cur_task = key
                        break

            if not cur_task:
                print(raw.event_id)
                print('\tNo match found for file.')

            file_labels.append((subj_file[0:9], subj_file, cur_task))

    # Save out a file of all the mappings
    with open('file_mappings.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for f_info in file_labels:
            writer.writerow(list(f_info))


if __name__ == '__main__':
    main()
