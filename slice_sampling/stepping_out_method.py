# Slice sampling
# Use of stepping out procedure
# paper: Slice Sampling by Radford M. Neal
# Author: Giorgio Sgarbi
from typing import Any, Union

import scipy.stats as stats
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
import math

t = sym.Symbol('t')

# return interval I = (l, r) using stepping out procedure
def find_interval(current_x: float, y: float, w: float = 0.5, m: int = 1000) -> tuple:
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
    candidate_sample: float = float(l + (r - l) * stats.uniform.rvs())
    candidate: float = float(f(candidate_sample))

    if candidate > y:  # base case: accept candidate
        return float(candidate_sample)
    else:  # use shrinking
        if candidate_sample < current_x:
            return (sample_from_slice(current_x, y,
                                      candidate_sample, r))
        else:
            return (sample_from_slice(current_x, y,
                                      l, candidate_sample))


# returns array of samples
# num_samples: number of desired samples
# x_0: initial sample
def slice_sampling(num_samples: int, x_0: float) -> ndarray:
    # check if density at initial point is positive
    assert f(x_0) > 0

    # initialize array with x_0
    array_samples: ndarray = np.array([x_0])

    for i in range(num_samples - 1):
        current_x = array_samples[-1]

        # sample next y: draw y uniformly from (0, f(current_x))
        y = float(stats.uniform.rvs(loc=0, scale=f(current_x)))

        # find interval I = (l,r)
        l, r = find_interval(current_x, y)
        # check that current_x is in (l,r)
        assert l < current_x < r

        # sample next x
        next_sample: float = sample_from_slice(current_x, y, l, r)

        # add next x to samples
        array_samples = np.append(array_samples, next_sample)

        print("SAMPLED x =", next_sample)

    return array_samples


# function to with the density of x is proportional
def f(x):
    y = (1 + np.sin(3*x)**2) * (1 + np.cos(5*x)**4) * np.exp(-x**2/2)
    return y


# render plot
def render_plot(samples):
    fig, ax = plt.subplots()

    num_bins = 80

    # the histogram of the data
    n, bins, patches = ax.hist(samples, num_bins, density=True,
                               alpha=0.75, linewidth=0.2)

    # add a 'best fit' line with norm constant = 5.16982
    y = f(bins)/5.16982
    ax.plot(bins, y, '-', color='orange', linewidth=3, label='normalized f(x)')
    ax.set_xlabel('Samples')
    ax.set_title(r"Slice Sampling with stepping out method. $x_0 = 0.5$. $50000$ samples")
    plt.text(x=-4, y=0.61, s=r'$f(x) = (1 + sin^2{3x}) (1 + cos^45x) e^{-\frac{x^2}{2}}$', weight=20, size=9)

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')

    # Tweak spacing to prevent clipping of y label
    fig.tight_layout()
    plt.savefig('plot_examples/stepping_out', dpi=None, facecolor='w',
                edgecolor='w', orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)


def main():
    print("SAMPLING STARTED")
    samples = slice_sampling(50000, 0.5)
    render_plot(samples)
    print("graph saved in 'plot_examples/stepping_out")


if __name__ == "__main__":
    main()
