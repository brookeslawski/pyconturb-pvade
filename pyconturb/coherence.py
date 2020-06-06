# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""
import itertools

import numpy as np


def get_coh_mat(freq, spat_df, coh_model='iec', dtype=np.float64, **kwargs):
    """Create coherence matrix for given frequencies and coherence model
    """
    if coh_model == 'iec':  # IEC coherence model
        if 'ed' not in kwargs.keys():  # add IEC ed to kwargs if not passed in
            kwargs['ed'] = 3
        coh_mat = get_iec_coh_mat(freq, spat_df, dtype=dtype, **kwargs)
    elif coh_model == '3d':  # 3D coherence model
        coh_mat = get_3d_coh_mat(freq, spat_df, dtype=dtype, **kwargs)

    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized.')

    return coh_mat

def chunker(iterable,nPerChunks):
    """ Return list of nPerChunks elements of an iterable """
    it = iter(iterable)
    while True:
       chunk = list(itertools.islice(it, nPerChunks))
       if not chunk:
           return
       yield chunk

def get_iec_coh_mat(freq, spat_df, dtype=np.float64, **kwargs):
    """Create IEC 61400-1 Ed. 3 coherence matrix for given frequencies
    """
    # preliminaries
    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    freq = np.array(freq).reshape(1, -1)
    n_f, n_s = freq.size, spat_df.shape[1]
    # misc storage
    xyz = spat_df.loc[['x', 'y', 'z']].values.astype(float)
    coh_mat = np.repeat(np.atleast_3d(np.eye((n_s),dtype=dtype)), n_f, axis=2) # TODO use sparse matrix 
    exp_constant = np.sqrt( (1/ kwargs['u_ref'] * freq)**2 + (0.12 / kwargs['l_c'])**2).astype(dtype)
    Icomp = np.arange(n_s)[spat_df.iloc[0, :].values==0]  # Selecting only u-components
    # loop through number of combinations, nPerChunks at a time to reduce memory impact
    for ii_jj in chunker(itertools.combinations(Icomp, 2), nPerChunks=10000):
        # get indices of point-pairs
        ii = np.array([tup[0] for tup in ii_jj])
        jj = np.array([tup[1] for tup in ii_jj])
        # calculate distances and coherences
        r = np.sqrt((xyz[1, ii] - xyz[1, jj])**2 + (xyz[2, ii] - xyz[2, jj])**2)
        coh_values = np.exp(-12 * np.abs(r.reshape(-1, 1)) * exp_constant)
        # Previous method (same math, different numerics)
        # coh_values = np.exp(-12 *
        #                     np.sqrt((r.reshape(-1, 1) / kwargs['u_ref'] * freq)**2
        #                             + (0.12 * r.reshape(-1, 1) / kwargs['l_c'])**2))
        coh_mat[ii, jj, :] = coh_values
        coh_mat[jj, ii, :] = np.conj(coh_values)
    return coh_mat


def get_3d_coh_mat(freq, spat_df, dtype=np.float64, **kwargs):
    """Create coherence matrix with 3d coherence for given frequencies
    """
    if any([k not in kwargs.keys() for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    # intermediate variables
    freq = np.array(freq).reshape(1, -1)
    n_f, n_s = freq.size, spat_df.shape[1]
    # misc storage
    xyz = spat_df.loc[['x', 'y', 'z']].values
    coh_mat = np.repeat(np.atleast_3d(np.eye((n_s),dtype=dtype)), n_f, axis=2)
    # loop through the three components
    for (k, lc_scale) in [(0, 1), (1, 2.7 / 8.1), (2, 0.66 / 8.1)]:
        Icomp = np.arange(n_s)[spat_df.iloc[0, :].values==k]  # Selecting only 1 component
        l_c = kwargs['l_c'] * lc_scale
        exp_constant = np.sqrt( (1/ kwargs['u_ref'] * freq)**2 + (0.12 / l_c)**2).astype(dtype)
        # loop through number of combinations, nPerChunks at a time to reduce memory impact
        for ii_jj in chunker(itertools.combinations(Icomp, 2), nPerChunks=10000):
            ii = np.array([tup[0] for tup in ii_jj])
            jj = np.array([tup[1] for tup in ii_jj])
            r = np.sqrt((xyz[1, ii] - xyz[1, jj])**2 + (xyz[2, ii] - xyz[2, jj])**2)
            coh_values = np.exp(-12 * np.abs(r.reshape(-1, 1)) * exp_constant)
            # Previous method (same math, different numerics)
            #coh_values = np.exp(-12 *
            #                    np.sqrt((r.reshape(-1, 1) / kwargs['u_ref'] * freq)**2
            #                            + (0.12 * r.reshape(-1, 1) / l_c)**2))
            coh_mat[ii, jj, :] = coh_values
            coh_mat[jj, ii, :] = np.conj(coh_values)
    return coh_mat
