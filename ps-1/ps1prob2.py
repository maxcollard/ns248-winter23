"""Helper functions for Problem Set 1 of UCSF NS248"""


#
# Imports
#

import numpy as np
import matplotlib.pyplot as plt

import ns248 as ns


#
# Globals
#

problem_names = {
    '1a': 'Problem 1a: $\operatorname{Pr}(A\mid C)$',
    '1b': 'Problem 1b: $\operatorname{Pr}(B\mid C)$',
    '1c': 'Problem 1c: $\operatorname{Pr}(A\cap B\mid C)$',
    '1d': 'Problem 1d: $\operatorname{Pr}(A\cap B^c\mid C)$',
    '1e': 'Problem 1e: $\operatorname{Pr}(B\cap A^c\mid C)$',
}

problem_keys = list( sorted( problem_names.keys() ) )


#
# Default parameters
#

# Probability of A and B independently firing
pA_default = 0.1
pB_default = 0.4

# Conditional probabilities for C
pC_cond_default = {
    (False, False): 0.,
    (True, False): 0.5,
    (False, True): 0.2,
    (True, True): 1.
}


#
# Functions
#

def simulate_loop( n,
                   pA = pA_default,
                   pB = pB_default,
                   pC_cond = pC_cond_default ):
    """Simulate `n` runs of Problem 2 with the given params (loop implementation)"""
    
    ret = {
        'A': [],
        'B': [],
        'C': [],
    }
    
    for i in range( n ):
        
        fired_a = ns.random_fire( pA )
        fired_b = ns.random_fire( pB )
        
        pC = pC_cond[(fired_a, fired_b)]
        fired_c = ns.random_fire( pC )
        
        ret['A'].append( fired_a )
        ret['B'].append( fired_b )
        ret['C'].append( fired_c )
    
    # For convenience, transform the return values into `numpy` arrays
    ret = { k: np.array( v )
            for k, v in ret.items() }
    
    return ret

def simulate_vectorized( n,
                         pA = pA_default,
                         pB = pB_default,
                         pC_cond = pC_cond_default ):
    """Simulate `n` runs of Problem 2 with the given params (vector implementation)
    
    pC_cond is encoded as a `dict` keyed as (A_fired?, B_fired?) -> pC
    """
    
    # Let's reformat the parameters into a form that can be quickly accessed
    # using the vector index math in `numpy`
    pC_cond_fast = np.array( [
        pC_cond[(False, False)],
        pC_cond[(True, False)],
        pC_cond[(False, True)],
        pC_cond[(True, True)],
    ] )
    
    # We use < and not <= because `np.random.uniform` usees half-open intervals
    # of the form [0, 1)
    fired_A = np.random.uniform( size = (n,) ) < pA
    fired_B = np.random.uniform( size = (n,) ) < pB
    
    # Determine the probability of C firing in each trial based on the
    # conditional dependencies
    cond_fast_idx = fired_A + 2 * fired_B # 0 is (False, False), 1 is (True, False), etc.
    pC_trials = pC_cond_fast[cond_fast_idx]
    fired_C = np.random.uniform( size = (n,) ) < pC_trials
    
    ret = {
        'A': fired_A,
        'B': fired_B,
        'C': fired_C,
    }
    
    return ret

# Give a way of accessing all implementations
simulators = {
    'loop': simulate_loop,
    'vectorized': simulate_vectorized,
}

def evaluate( data ):
    """Evaluate Problem 2 simulation results to compute the values for each question"""
    
    # Our return value
    ret = dict()
    
    # 1a: Pr( A | C )
    ret['1a'] = ns.pr( data['A'], given = data['C'] )
    # 1b: Pr( B | C )
    ret['1b'] = ns.pr( data['B'], given = data['C'] )
    # 1c: Pr( A & B | C )
    ret['1c'] = ns.pr( data['A'] & data['B'], given = data['C'] )
    # 1d: Pr( A & not-B | C )
    ret['1d'] = ns.pr( data['A'] & ~data['B'], given = data['C'] )
    # 1e: Pr( B & not-A | C )
    ret['1e'] = ns.pr( data['B'] & ~data['A'], given = data['C'] )
    
    return ret

def plot( ax, results,
          quantiles = None,
          theoretical = None,
          style = 'single' ):
    """Plot results for one part of Problem 2
    
    Results is a `dict` of the form {n: [r0, ... rn]} where `n` is the number
    of simulation runs used, and each `ri` is the result for one batch
    
    `style` is either 'single' for one simulation (default), or 'mean' for mean across simulations
    """
    
    if quantiles is None:
        # Use reasonable defaults
        quantiles = [
            0.025,
            0.25,
            0.75,
            0.975,
        ]
    
    # Extract the n-values used in results
    ns = np.array( list( sorted( results.keys() ) ) )
    
    if style == 'single':
        i_plot = np.random.randint( len( results[ns[0]] ) )
        plot_values = np.array( [ results[n][i_plot]
                                  for n in ns ] )
        plot_legend = 'Chosen replicate'
    elif style == 'mean':
        # We use the `nan` versions of these functions because conditional probability
        # might be undefined in some cases, and we want to ignore these trials
        plot_values = np.array( [ np.nanmean( results[n] )
                                  for n in ns ] )
        plot_legend = 'Mean of replicates'
    else:
        raise ValueError( f'Unknown plot style: {style}' )
    
    ax.semilogx( ns, plot_values, 'k-',
                 label = plot_legend )
    
    if theoretical is not None:
        ax.semilogx( ns, theoretical * np.ones( ns.shape ), 'k--',
                     linewidth = 1,
                     label = 'Theoretical' )
    
    # `enumerate` gives back both the index and the list value itself
    for i, q in enumerate( quantiles ):
        quantile_values = np.array( [ np.nanquantile( results[n], q )
                                      for n in ns ] )
        
        ax.semilogx( ns, quantile_values, f'C{i}--',
                     label = f'{q * 100:0.1f} %ile' )