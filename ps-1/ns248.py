"""General helper functions for UCSF NS248"""

# Imports

import numpy as np
import pandas as pd

# Functions

def append_by_keys( acc, d ):
    """Append a dict of values `d` onto an accumulator dict `acc` *by reference*
    
    `acc` is of form {k: [v0, ... vn]}, `d` is of form {k: vNew}
    Thus function returns a `dict` of the form {k: [v0, ... vn, vNew]}
    """
    
    if acc is None:
        return { k: [v]
                 for k, v in d.items() }
    
    for k in acc.keys():
        acc[k].append( d[k] )
    return acc

# 

def random_fire( p ):
    """A r.v. that is True with probability `p`"""
    if np.random.uniform() < p:
        return True
    else:
        return False

def pr( xs, given = None ):
    """The probability that `xs` is true
    
    If `given` is not None, condition on `given`.
    """
    
    if xs.size == 0:
        # Probability on an empty space is undefined
        return np.nan
    
    if given is None:
        return np.sum( xs == True ) / xs.size
    
    # `given` is now not None
    
    if xs[given].size == 0:
        # Conditional probability conditioned on a set of measure zero is
        # undefined in the discrete case
        return np.nan
    
    return np.sum( xs[given] == True ) / xs[given].size

def edges( bin_edges ):
    """Given an edge list, return an iterator over the (left, right) edges of each bin"""
    return zip( bin_edges[:-1], bin_edges[1:] )

def centers( bin_edges ):
    """Given an edge list, return an array with the center of each bin"""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])

