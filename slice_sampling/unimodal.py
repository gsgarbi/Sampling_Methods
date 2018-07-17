# Slice sampling for unimodal distributions
# paper: Slice Sampling (by Radford M. Neal)
# Author: Giorgio Sgarbi

import scipy.stats as stats
import numpy as np
import sympy as sym
t = sym.Symbol('t')

import matplotlib.pyplot as plt
import pandas as pd
import csv

# returns array of samples
# num_samples: number of desired samples
# x_0: initial sample
def slice_sampling(num_samples, x_0):
    #check if density at initial point is positive
    assert f(x_0) > 0
    
    # initialize array with x_0
    array_samples = np.array([x_0])

    for i in range(num_samples - 1):
        # draw next sample from f(x)
        next_sample = sample_next(array_samples)
        
        ## add nextx to samples 
        ## (note for developers: np.append does NOT occur in place)
        array_samples = np.append(array_samples, next_sample)
        
    return (array_samples)

# returns next sample
# array_sample: current array of samples already drawn
def sample_next(array_samples):
    # previous sample is the last sample in the array of samples
    previous_sample = array_samples[-1]
    
    # draw y uniformly from (0, f(previous_sample))
    y = stats.uniform.rvs(loc = 0, scale = f(previous_sample))
    
    # Assuming it is possible, find intersections of g(x) = y and f(x)
    intersections = sym.solve(f(t) - y, t)

    # find interval I = (L,R) for unimodal
    assert len(intersections) == 2
    [L,R] = np.asanyarray(intersections, dtype = float)
    
    # draw next_sample uniformly(?) from I
    next_sample = L + (R - L) * stats.uniform.rvs()
    
    return (next_sample)


# function
# use sym notation or np notation
## note for developer: if, for instance, np.exp is used, 
## sym will not recognize it as a valid operation    
def f(x, mu = 0, sigma = 1, NP = False):
    if NP:
        y = (1 / (np.sqrt(2 * np.pi) * sigma) *
         np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
    else:
        y = (1 / (sym.sqrt(2 * sym.pi) * sigma) *
         sym.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))

    return (y)


# render plot
def render_plot(samples):
    fig, ax = plt.subplots()
    
    num_bins = 80
    
    # the histogram of the data
    n, bins, patches = ax.hist(samples, num_bins, density = True, 
                               alpha = 0.75, linewidth = 0.2)
 
    # add a 'best fit' line
    mu, sigma = 0, 1 
    y = f(bins, mu, sigma, True)
    ax.plot(bins, y, '-', color = 'orange', linewidth = 3)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Normal({}, {})'.format(mu, sigma))
    
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig('plot_examples/N01_5000', dpi=None, facecolor='w', 
            edgecolor='w',orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None)
    plt.show()
    

    

# uncomment once to test sampling_method function
test_code = slice_sampling(50, 0.2)
print (test_code)
render_plot(test_code)

## uncomment once to test sampling method by plotting
## create custom experiment or use example from N(1,0) with initial point 0.2
## slice_sampling(5000, 0.2) saved in 'N01_5000_0.2.csv'
samples = slice_sampling(5000, 0.2)


#df=pd.read_csv('data_examples/N01_5000_0.2.csv', sep=',',header=None)
#samples = df.values.T
#render_plot(samples)