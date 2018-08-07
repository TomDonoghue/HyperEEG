"""Analysis functions for HyperEEG project."""

import numpy as np
from sklearn.model_selection import cross_val_score

from settings import K_FOLD

###################################################################################################
###################################################################################################

# Initialize an SVM classification object
#CLF = svm.SVC(kernel='linear')
CLF = svm.LinearSVC()

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


def maxabs(dat, dim):
    """   """

    return np.max(np.abs(dat), dim)
