# Slice sampling for bimodal distribution: 0.2N(-10, 1) + 0.8N(10, 1)
# paper: Slice Sampling by Radford M. Neal
# Author: Giorgio Sgarbi
from typing import Any, Union

import scipy.stats as stats
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray

t = sym.Symbol('t')


# returns array of samples
# num_samples: number of desired samples
# x_0: initial sample
def slice_sampling(num_samples: int, x_0: int) -> ndarray:
    # check if density at initial point is positive
    assert f(x_0) > 0

    # initialize array with x_0
    array_samples: ndarray = np.array([x_0])

    for i in range(num_samples - 1):
        previous_sample = array_samples[-1]

        # sample next y: draw y uniformly from (0, f(previous_sample))
        y = float(stats.uniform.rvs(loc=0, scale=f(previous_sample)))

        # find interval
        [L, R] = find_interval(y)
        # check that previous_sample is in (L,R)
        assert L < previous_sample < R

        # sample next x
        next_sample: float = sample_from_slice(previous_sample, y, L, R)

        # add next x to samples
        # (note for self: np.append does NOT occur in place)
        array_samples = np.append(array_samples, next_sample)

    return array_samples


# return I such that I = (inf(S), sup(S))
def find_interval(y: float) ->list:
    # Assuming it is possible, find intersections of g(x) = y and f1(x)
    intersections1 = sym.solve(f1(t) - y, t)
    # find interval I = (L,R) for unimodal f1
    assert len(intersections1) == 2
    [L1, R1] = np.asanyarray(intersections1, dtype=float)

    # Assuming it is possible, find intersections of g(x) = y and f2(x)
    intersections2 = sym.solve(f2(t) - y, t)
    # find interval I = (L,R) for unimodal f2
    assert len(intersections2) == 2
    [L2, R2] = np.asanyarray(intersections2, dtype=float)

    return [L1, R2]


# S = {x: x in I and f(x) > y}
def sample_from_slice(previous_sample, y, L, R):
    candidate_sample = L + (R - L) * stats.uniform.rvs()
    f_candidate = float(f(candidate_sample))

    if f_candidate > y:  # base case
        return float(candidate_sample)
    else:
        if candidate_sample < previous_sample:
            return (sample_from_slice(previous_sample, y,
                                  candidate_sample, R))
        else:
            return (sample_from_slice(previous_sample, y,
                                  L, candidate_sample))


# function
# use sym notation or np notation
# note for developer: if, for instance, np.exp is used,
# sym will not recognize it as a valid operation
def f1(x, mu1=-10, sigma1=1, NP=False):
    if NP:
        y1 = (1 / (np.sqrt(2 * np.pi) * sigma1) *
              np.exp(-0.5 * (1 / sigma1 * (x - mu1)) ** 2))
    else:
        y1 = (1 / (sym.sqrt(2 * sym.pi) * sigma1) *
              sym.exp(-0.5 * (1 / sigma1 * (x - mu1)) ** 2))

    return y1


def f2(x, mu2=10, sigma2=1, NP=False):
    if NP:
        y2 = (1 / (np.sqrt(2 * np.pi) * sigma2) *
              np.exp(-0.5 * (1 / sigma2 * (x - mu2)) ** 2))
    else:
        y2 = (1 / (sym.sqrt(2 * sym.pi) * sigma2) *
              sym.exp(-0.5 * (1 / sigma2 * (x - mu2)) ** 2))

    return y2


def f(x, mu1=-10, sigma1=1, mu2=10, sigma2=1, NP=False):
    if NP:
        y1 = (1 / (np.sqrt(2 * np.pi) * sigma1) *
              np.exp(-0.5 * (1 / sigma1 * (x - mu1)) ** 2))
        y2 = (1 / (np.sqrt(2 * np.pi) * sigma2) *
              np.exp(-0.5 * (1 / sigma2 * (x - mu2)) ** 2))
    else:
        y1 = (1 / (sym.sqrt(2 * sym.pi) * sigma1) *
              sym.exp(-0.5 * (1 / sigma1 * (x - mu1)) ** 2))
        y2 = (1 / (sym.sqrt(2 * sym.pi) * sigma2) *
              sym.exp(-0.5 * (1 / sigma2 * (x - mu2)) ** 2))

    return 0.2 * y1 + 0.8 * y2


# render plot
def render_plot(samples):
    fig, ax = plt.subplots()

    num_bins = 80

    # the histogram of the data
    n, bins, patches = ax.hist(samples, num_bins, density=True,
                               alpha=0.75, linewidth=0.2)

    # add a 'best fit' line
    y = f(bins, NP=True)
    ax.plot(bins, y, '-', color='orange', linewidth=3)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Normal')

    # Tweak spacing to prevent clipping of y label
    fig.tight_layout()
    plt.savefig('plot_examples/Normal_mixture', dpi=None, facecolor='w',
                edgecolor='w', orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.show()


def main():
    print("SAMPLING STARTED")
    samples = slice_sampling(5000, 10.1)
    render_plot(samples)


if __name__ == "__main__":
    main()

