# Slice sampling
# Use of stepping out procedure
# paper: Slice Sampling by Radford M. Neal
# Author: Giorgio Sgarbi

import math
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray


def slice_sampling(num_samples: int, x_0: float) -> ndarray:
    """ Initiate the Slice sampling method
    :param num_samples: int representing the number of desired samples
    :param x_0: float representing an initial sample with f(x_0) > 0
    :return: array of samples following normalized f(x)
    """
    # check if density at initial point is positive
    assert f(x_0) > 0

    array_samples: ndarray = np.array([x_0])

    for i in range(num_samples - 1):
        current_x = array_samples[-1]

        # sample next y: draw y uniformly from (0, f(current_x))
        y = float(stats.uniform.rvs(loc=0, scale=f(current_x)))

        # find interval I = (l,r)
        l, r = find_interval(current_x, y)

        # check that current_x is in (l,r)
        assert l < current_x < r

        # sample next x: draw x uniformly from slice S
        next_sample: float = sample_from_slice(current_x, y, l, r)

        # add next x to set of samples
        array_samples = np.append(array_samples, next_sample)

    return array_samples


def find_interval(current_x: float, y: float, w: float = 1.5, m: int = 10000) -> tuple:
    """ Stepping out method
    :param current_x: current value of x
    :param y: current value of y
    :param w: estimate of the size of a slice
    :param m: integer to limit the size of a slice to m * w
    :return: tuple (l,r) representing the size of the next interval
    """
    u: float = float(stats.uniform.rvs(loc=0, scale=1))
    l: float = current_x - w * u
    r = l + w
    v = float(stats.uniform.rvs(loc=0, scale=1))
    j = math.floor(m * v)
    k = m - 1 - j

    while j > 0 and y < f(l):
        l -= w
        j -= 1
    while k > 0 and y < f(r):
        r += w
        k -= 1
    return l, r


# sample from slice S = {x: x in I and f(x) > y} using shrinking
def sample_from_slice(current_x, y, l, r):
    """ Sample uniformly from slice (l,r)
    :param current_x: current value of x
    :param y: current value of y
    :param l: left limit of interval
    :param r: right limit of interval
    :return: a sample drawn from a distribution proportional to f(x)
    """
    candidate_sample: float(l + (r - l) * stats.uniform.rvs())
    candidate = float(f(candidate_sample))

    if candidate > y:  # base case: accept candidate
        return float(candidate_sample)
    else:  # use shrinking
        if candidate_sample < current_x:
            return (sample_from_slice(current_x, y,
                                      candidate_sample, r))
        else:
            return (sample_from_slice(current_x, y,
                                      l, candidate_sample))


# function to which the density of x is proportional
def f(x):
    # f is not necessarily normalized
    y = (1 + np.sin(3 * x) ** 2) * (1 + np.cos(5 * x) ** 4) * np.exp(-x ** 2 / 2)
    return y


# render boxplot with
def render_plot(samples):
    fig, ax = plt.subplots()

    num_bins = 80

    # the histogram of the data
    n, bins, patches = ax.hist(samples, num_bins, density=True,
                               alpha=0.75, linewidth=0.2)

    # add a 'best fit' line for f(x) with norm constant Z = 5.16982
    z = 5.16982
    y = f(bins) / z
    ax.plot(bins, y, '-', color='orange', linewidth=3, label='normalized f(x)')
    ax.set_xlabel('Samples')
    ax.set_title(r"Slice Sampling with stepping out method. $x_0 = 0.5$. $50000$ samples")
    plt.text(x=-4, y=0.61, s=r'$f(x) = (1 + sin^2{3x}) (1 + cos^45x) e^{-\frac{x^2}{2}}$',
             weight=20, size=9)
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')

    # Tweak spacing
    fig.tight_layout()
    plt.savefig('plot_examples/stepping_out', dpi=None, facecolor='w',
                edgecolor='w', orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)


if __name__ == "__main__":
    print("SAMPLING STARTED")
    samples = slice_sampling(50000, 0.5)
    render_plot(samples)
    print("graph saved in 'plot_examples/stepping_out")
