"""Settings for HyperEEG project."""

from analysis import maxabs

##
##

# Data size
N_EPOCHS = 40
N_CHS = 128
N_PER_COND = int(n_epochs / 2)

# Set the collection of ways to average across features
#  Note: these are used
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
