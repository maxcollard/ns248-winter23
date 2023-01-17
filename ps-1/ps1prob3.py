""""""


#
# Imports
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ns248 as ns


#
# Functions
#

def plot_hists( ax, data,
                value_key = 'spikes',
                group_key = 'stimulus',
                bin_width = None,
                groups = None,
                alpha = None ):
    """Plot histograms of the `value_key` column of `data` inside of `ax`, stratified by the categories in `group_key`
    
    If `bin_width` is not specified, use integral bins
    """
    
    if groups is None:
        # Use all the stimuli that are in the data.
        # There are other ways to get at this with `pandas`, but this ensures
        # the groups are sorted
        groups = list( sorted( data[group_key].unique() ) )
    
    if alpha is None:
        alpha = 1. / len( groups )
    
    if bin_width is None:
        # We want the bins to be centered on integer values from zero up to the max
        bin_width = 1.
        bin_edges = np.arange( 0, np.max( data[value_key] ) + bin_width, bin_width ) - (bin_width / 2)
    else:
        # If we specify a width, don't center the same way: just space them out appropriately
        bin_edges = np.arange( 0, np.max( data[value_key] ) + bin_width, bin_width )
    
    # Plot the histogram for each group
    for i_group, group in enumerate( groups ):
        
        # Filter out the trials for this group
        filter_cur = data[group_key] == group
        data_cur = data[filter_cur]
        
        # Manually pull out the histogram counts from `numpy`
        values_cur = data_cur[value_key]
        hist_counts, _ = np.histogram( values_cur, bins = bin_edges )
        
        # TODO There is definitely a faster way to plot these but I have no motivation
        # to make my plotting more efficient 😬
        for i, (bin_center, bin_count) in enumerate( zip( ns.centers( bin_edges ), hist_counts ) ):
            ax.bar( bin_center, bin_count,
                    width = bin_width,
                    color = f'C{i_group % 10}',
                    alpha = alpha,
                    label = f'{group_key} = {group}' if i == 0 else None )
    
    ax.set_xlabel( value_key )
    ax.set_ylabel( 'Count' )
    ax.legend()

def pmfs( data,
          bin_width = None,
          value_key = 'spikes',
          group_key = 'stimulus',
          groups = None ):
    """Determine the pmfs of the `value_key` column of `data`, stratified by `group_key`
    
    If `bin_width` is not specified, use integral values (the empirical, unbinned pmf)
    """
    
    if groups is None:
        groups = list( sorted( data[group_key].unique() ) )
    
    if bin_width is None:
        bin_edges = np.arange( 0, np.max( data[value_key] ) + 1, 1 ) - (1 / 2)
    else:
        bin_edges = np.arange( 0, np.max( data[value_key] ) + bin_width, bin_width )
    
    ret_data = {
        f'low::{value_key}': bin_edges[:-1],
        f'high::{value_key}': bin_edges[1:],
    }
    if bin_width is None:
        ret_data[f'center::{value_key}'] = ns.centers( bin_edges ).astype( int )
    else:
        ret_data[f'center::{value_key}'] = ns.centers( bin_edges )
    
    for i_group, group in enumerate( groups ):
        
        # Filter out the trials for this group
        filter_cur = data[group_key] == group
        data_cur = data[filter_cur]

        # Let's just use `numpy` to do the histogram counts to make a pmf
        values_cur = data_cur[value_key]
        hist_counts, _ = np.histogram( values_cur, bins = bin_edges )
        pmf_cur = hist_counts / np.sum( hist_counts )
        
        key_cur = f'pmf::{group_key}::{group}'
        ret_data[key_cur] = pmf_cur
    
    return pd.DataFrame( ret_data )

def parse_pmf_columns( pmf_data ):
    """Use the `DataFrame` formatting from `pmfs` to determine which column is which"""
    
    group_columns = list( sorted( [ k
                                    for k in pmf_data.keys()
                                    if 'pmf::' in k ] ) )
    
    value_columns = list( sorted( [ k
                                    for k in pmf_data.keys()
                                    if 'center::' in k ] ) )
    
    other_columns = [ k
                      for k in pmf_data.keys()
                      if k not in group_columns + value_columns ]
        
    return group_columns, value_columns, other_columns

def cdfs( data, xs,
          value_key = 'spikes',
          group_key = 'stimulus',
          groups = None ):
    """Determine the cdfs of the `value_key` column of `data` evaluated at `xs, stratified by `group_key`"""
    
    if groups is None:
        groups = list( sorted( data[group_key].unique() ) )
    
    ret_data = {
        f'value::{value_key}': xs,
    }
    
    for i_group, group in enumerate( groups ):
        
        # Filter out the trials for this group
        filter_cur = data[group_key] == group
        data_cur = data[filter_cur]

        # Use the definition of cdf
        values_cur = data_cur[value_key]
        cdf_cur = np.array( [ np.sum( values_cur <= x ) / len( values_cur )
                              for x in xs ] )
        
        key_cur = f'cdf::{group_key}::{group}'
        ret_data[key_cur] = cdf_cur
    
    return pd.DataFrame( ret_data )

def parse_cdf_columns( pmf_data ):
    """Use the `DataFrame` formatting from `cdfs` to determine which column is which"""
    
    group_columns = list( sorted( [ k
                                    for k in pmf_data.keys()
                                    if 'cdf::' in k ] ) )
    
    value_columns = list( sorted( [ k
                                    for k in pmf_data.keys()
                                    if 'value::' in k ] ) )
    
    other_columns = [ k
                      for k in pmf_data.keys()
                      if k not in group_columns + value_columns ]
        
    return group_columns, value_columns, other_columns

def plot_pmfs( ax, pmf_data ):
    """Plot the pmfs created as output from `pmfs` inside of `ax`"""
    
    group_columns, value_columns, other_columns = parse_pmf_columns( pmf_data )
    
    if len( value_columns ) > 1:
        raise ValueError( 'Unsure which column is the value column' )
    if len( value_columns ) == 0:
        raise ValueError( 'No value columns' )
    value_column = value_columns[0]
    value_key = value_column.split( '::' )[-1]
    
    for i_row, row in pmf_data.iterrows():
        
        x_value = row[value_column]
        
        for i_group, group in enumerate( group_columns ):
            group_disp = '::'.join( group.split( '::' )[1:] )
            
            pmf_value = row[group]
            
            ax.stem( x_value + i_group * 0.2, pmf_value, f'C{i_group % 10}',
                     label = group_disp if i_row == 0 else None )

    ax.set_xlabel( value_key )
    ax.set_ylabel( 'Probability' )
    ax.legend()
    
def plot_cdfs( ax, cdf_data ):
    """Plot the cdfs created as output from `cdfs` inside of `ax`"""
    
    group_columns, value_columns, other_columns = parse_cdf_columns( cdf_data )
    
    if len( value_columns ) > 1:
        raise ValueError( 'Unsure which column is the value column' )
    if len( value_columns ) == 0:
        raise ValueError( 'No value columns' )
    value_column = value_columns[0]
    value_key = value_column.split( '::' )[-1]
    
    values = cdf_data[value_column]
    
    for i_group, group in enumerate( group_columns ):
        group_disp = '::'.join( group.split( '::' )[1:] )
        cur_cdf = cdf_data[group]

        ax.plot( values, cur_cdf, f'C{i_group%10}-',
                 label = group_disp )

    ax.set_xlabel( value_key )
    ax.set_ylabel( 'Cumulative probability' )
    ax.legend()
    
def uniform_prior( group_columns ):
    """Return a uniform distribution over `group_columns`"""
    n = len( group_columns )
    return { group: 1. / n
             for group in group_columns }

def bayes( pmf_data,
           prior = None ):
    """Apply Bayes' rule to pmfs from `pmfs` with the given `prior` (defaultl: uniform)"""
    
    group_columns_pmf, value_columns, other_columns = parse_pmf_columns( pmf_data )
    copy_columns = value_columns + other_columns
    
    transform_group = lambda x: '::'.join( ['posterior'] + x.split( '::' )[1:] )
    
    if prior is None:
        # Since no prior is given, assume an "uninformative" prior
        prior = uniform_prior( group_columns_pmf )
        
    # Make sure that prior is properly normalized
    prior_sum = np.sum( [ x for _, x in prior.items() ] )
    prior = { k: v / prior_sum for k, v in prior.items() }
    
    ret_data = dict()
    for column in copy_columns:
        ret_data[column] = pmf_data[column]
    for group in group_columns_pmf:
        ret_data[transform_group(group)] = []
    
    for i_row, row in pmf_data.iterrows():
        
        # `row` holds the conditional probability of each group given that
        # we observed `row[value_column]` to be a particular value
        
        # Use the law of total probability for the marginal
        denominator = np.sum( [ row[group] * prior[group]
                                for group in group_columns_pmf ] )
        
        if denominator == 0:
            # Bayes' rule is ill-defined when the marginal is zero
            posterior_cur = { group: np.nan
                              for group in group_columns_pmf }
        else:
            # ** This is the Bayes' rule step! **
            posterior_cur = { group: row[group] * prior[group] / denominator
                              for group in group_columns_pmf }
        
        for group in group_columns_pmf:
            ret_data[transform_group(group)].append( posterior_cur[group] )
    
    return pd.DataFrame( ret_data )

def parse_posterior_columns( pmf_data ):
    """Use the `DataFrame` formatting from `bayes` to determine which column is which"""
    
    group_columns = list( sorted( [ k
                                    for k in pmf_data.keys()
                                    if 'posterior::' in k ] ) )
    
    non_group_columns = [ k
                          for k in pmf_data.keys()
                          if k not in group_columns ]
    
    left_columns = [ k for k in non_group_columns if 'low::' in k ]
    if len( left_columns ) > 1:
        raise ValueError( 'Unsure which column is the left value column' )
    if len( left_columns ) == 0:
        raise ValueError( 'No left value columns' )
    left_column = left_columns[0]
        
    right_columns = [ k for k in non_group_columns if 'high::' in k ]
    if len( right_columns ) > 1:
        raise ValueError( 'Unsure which column is the right value column' )
    if len( right_columns ) == 0:
        raise ValueError( 'No right value columns' )
    right_column = right_columns[0]
    
    other_columns = [ k
                      for k in pmf_data.keys()
                      if k not in group_columns + [left_column, right_column] ]
        
    return group_columns, left_column, right_column, other_columns

def plot_posterior( ax, posterior_data,
                    alpha = None ):
    """Plot the posterior distributions from `bayes` inside of `ax`"""
    
    group_columns, left_column, right_column, _ = parse_posterior_columns( posterior_data )
    
    if alpha is None:
        alpha = 1. / len( group_columns )
    
    value_key = left_column.split( '::' )[-1]
    
    transform_group_display = lambda x: '::'.join( x.split( '::' )[1:] )
    
    for i_row, row in posterior_data.iterrows():
        
        left_cur = row[left_column]
        right_cur = row[right_column]
        
        center_cur = (left_cur + right_cur) * 0.5
        width_cur = right_cur - left_cur
        
        for i_group, group in enumerate( group_columns ):
            posterior_value = row[group]
            
            ax.bar( center_cur, posterior_value,
                    color = f'C{i_group % 10}',
                    width = width_cur,
                    alpha = alpha,
                    label = transform_group_display( group ) if i_row == 0 else None )

    ax.set_xlabel( value_key )
    ax.set_ylabel( 'Posterior probability' )
    ax.legend()
    
    ax.set_xlabel( value_key )
    
def posterior_estimator_binned( posterior_data ):
    """Use an empirical posterior from `bayes` to construct an estimator of the posterior using binning
    
    Output is a function f(x: float) -> dict, with the dict representing the distribution over labels
    """
    
    group_columns, left_column, right_column, _ = parse_posterior_columns( posterior_data )
    
    transform_group_display = lambda x: '::'.join( x.split( '::' )[1:] )
    group_columns_display = [ transform_group_display( group )
                              for group in group_columns ]
    
    def f( x ):
        filter_left = x >= posterior_data[left_column]
        filter_right = x < posterior_data[right_column]
        
        df_intersection = posterior_data[filter_left & filter_right]
        if df_intersection.shape[0] == 0:
            raise ValueError( 'Input value is outside the domain of definition of the posterior' )
        if df_intersection.shape[0] > 1:
            raise ValueError( 'Posterior has two definitions for the input value' )
        row_domain = df_intersection.iloc[0]
        
        return { group_display: row_domain[group]
                 for group, group_display in zip( group_columns, group_columns_display ) }
    
    return f


#
# Classes
#

class EmpiricalBayesClassifier( object ):
    """Performs 1D classification by applying Bayes' rule to the empirical pmf"""
    
    def __init__( self,
                  prior = None,
                  bin_width = None ):
        """Create an estimator for the given `prior` and `bin_width` for discretizing data"""
        
        self.prior = prior
        self.bin_width = bin_width
        
        self._is_fit = False
        self.value_key = None
        self.pmfs = None
        self.posterior = None
        self.estimator = None
    
    def fit( self,
             X = None, y = None,
             data = None, value_key = None, class_key = None ):
        """Fit the estimator
        
        Uses either:
        * `X`, `y` as arrays, or
        * `data` as a `DataFrame`, with specified `value_key` and `class_key`
        """
        
        if X is not None and y is not None:
            data = pd.DataFrame( np.c_[X, y],
                                 columns = ['X', 'y'] )
            value_key = 'X'
            class_key = 'y'
        elif data is not None and value_key is not None and class_key is not None:
            pass
        else:
            raise ValueError( 'Specified data cannot fit model' )
        
        self.value_key = value_key
        
        self.pmfs = pmfs( data,
                          bin_width = self.bin_width,
                          value_key = value_key,
                          group_key = class_key )

        self.posterior = bayes( self.pmfs,
                                prior = self.prior )
        
        self.estimator = posterior_estimator_binned( self.posterior )
    
    def predict( self, X ):
        """Predict class probabilities based on values in `X`"""
        
        if type( X ) == pd.DataFrame:
            X_list = X[value_key]
        else:
            X_list = X
        
        # TODO Vectorize the estimator
        ret_data = None
        for x in X_list:
            ret_data = ns.append_by_keys( ret_data, self.estimator( x ) )
        
        return pd.DataFrame( ret_data )