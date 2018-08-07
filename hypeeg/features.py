"""Functions and helpers for selecting features for classification."""

import numpy as np

###################################################################################################
###################################################################################################

def maxabs(dat, dim):
    """Take the maximum of the absolute value of the signal."""

    return np.max(np.abs(dat), dim)

# Set the collection of ways to average across features
#  Note: these are used
AVGS = {
    'maxabs' : maxabs,
    'max' : np.max,
    'min' : np.min,
    'mean' : np.mean,
    'median' : np.median
}
