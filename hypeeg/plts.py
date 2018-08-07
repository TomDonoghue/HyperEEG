"""Plots for HyperEEG project."""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

###################################################################################################
###################################################################################################

PLTS_PATH = '/Users/tom/Desktop/HyperEEG_Project/Figures/'

###################################################################################################
###################################################################################################

def plot_results(times, results, labels, save_fig=False, save_name=None):
    """Plot

    Parameters
    ----------
    times : 1d array
        xx
    results : list of 1d arrays
        xx
    labels :
        xx
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    for result in results:
        plot_time_class(times, np.mean(result, 0), sem(result), ax)

    # Chance line @ 50%
    plot_time_class(times, np.ones(len(times)) * 0.5, ax=ax, color='grey', alpha=0.7)

    ax.set_xlim([-1, 1])
    ax.set_ylim([0.4, 0.9])

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14)

    ## Aesthetics

    #ax.grid(True)

    # Set tick fontsizes
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # Set the top and right side frame & ticks off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set linewidth of remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Add a legend
    plt.legend(labels, fontsize=14)

    if save_fig:
        save_name = os.path.join(PLTS_PATH, save_name + '.pdf')
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


def plot_time_class(times, dat, err=None, ax=None, *args, **kwargs):
    """

    Parameters
    ----------
    times :
        xx
    dat :
        xx
    err :
        xx
    ax :
        xx
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(times, dat, *args, **kwargs)

    if np.any(err):
        ax.fill_between(times, dat-err, dat+err, alpha=0.5)
