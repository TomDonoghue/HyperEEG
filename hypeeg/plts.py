"""Plots for HyperEEG project."""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

###################################################################################################
###################################################################################################

PLTS_PATH = '/Users/tom/Documents/Research/1-Projects/HyperEEG/4-Figures/'

###################################################################################################
###################################################################################################

def plot_results(times, results, labels, save_fig=False, save_name=None):
    """Plot

    Parameters
    ----------
    times : 1d array
        Vector of time values, to plot of the x-axis.
    results : list of 1d arrays
        List of data vectors to plot on the y-axis.
    labels : list of string
        Labels to use in the legend. Must have same length as results.
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
    """Plot a

    Parameters
    ----------
    times : 1d vector
        Vector of time values, to plot on the x-axis.
    dat : 1d vector
        Vector of data values, to plot on the y-axis.
    err : 1d vector
        Vector of errors per data value, to shade in.
    ax : matplotlib axes object, optional
        Axes object to plot on. If None, creates a new axis.

    Notes
    -----
    This is mostly a sub-function to plot each line in 'plot_results.'
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(times, dat, *args, **kwargs)

    if np.any(err):
        ax.fill_between(times, dat-err, dat+err, alpha=0.5)
