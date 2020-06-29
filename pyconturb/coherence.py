# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""
import itertools

import numpy as np


def get_coh_mat(freq, spat_df, coh_model='iec', dtype=np.float64, **kwargs):
    """Create coherence matrix for given frequencies and coherence model
    """
    if coh_model == 'iec':  # IEC coherence model
        coh_mat = get_iec_coh_mat(freq, spat_df, dtype=dtype, coh_model=coh_model,
                                  **kwargs)
    elif coh_model == 'iec3d':
        coh_mat = get_iec3d_coh_mat(freq, spat_df, dtype=dtype, coh_model=coh_model,
                                    **kwargs)
    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized!')

    return coh_mat


def chunker(iterable, nPerChunks):
    """Return list of nPerChunks elements of an iterable"""
    it = iter(iterable)
    while True:
       chunk = list(itertools.islice(it, nPerChunks))
       if not chunk:
           return
       yield chunk


def get_iec_coh_mat(freq, spat_df, dtype=np.float64, ed=4, **kwargs):
    """Create IEC 61400-1 Ed. 3/4 coherence matrix for given frequencies
    """
    # preliminaries
    if ed not in [3, 4]:  # only allow edition 3
        raise ValueError('Only editions 3 or 4 are permitted.')
    if any([k not in kwargs for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    freq = np.array(freq).reshape(-1, 1)  # nf x 1
    nf, ns = freq.size, spat_df.shape[1]
    # intermediate variables
    yz = spat_df.loc[['y', 'z']].values.astype(float)
    coh_mat = np.repeat(np.eye((ns), dtype=dtype)[None, :, :], nf, axis=0) # TODO use sparse matrix 
    exp_constant = np.sqrt( (1/ kwargs['u_ref'] * freq)**2 + (0.12 / kwargs['l_c'])**2).astype(dtype)  # nf x 1
    i_comp = np.arange(ns)[spat_df.iloc[0, :].values == 0]  # Selecting only u-components
    # loop through number of combinations, nPerChunks at a time to reduce memory impact
    for ii_jj in chunker(itertools.combinations(i_comp, 2), nPerChunks=10000):
        # get indices of point-pairs
        ii = np.array([tup[0] for tup in ii_jj])
        jj = np.array([tup[1] for tup in ii_jj])
        # calculate distances and coherences
        r = np.sqrt((yz[0, ii] - yz[0, jj])**2 + (yz[1, ii] - yz[1, jj])**2)  # nchunk
        coh_values = np.exp(-12 * np.abs(r) * exp_constant)  # nf x nchunk
        # Previous method (same math, different numerics)
        coh_mat[:, ii, jj] = coh_values  # nf x nchunk
        coh_mat[:, jj, ii] = np.conj(coh_values)
    return coh_mat


def get_iec3d_coh_mat(freq, spat_df, dtype=np.float64, **kwargs):
    """Create IEC-flavor coherence but for all 3 components
    """
    # preliminaries
    if any([k not in kwargs for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    freq = np.array(freq).reshape(-1, 1)  # nf x 1
    nf, ns = freq.size, spat_df.shape[1]
    # intermediate variables
    yz = spat_df.loc[['y', 'z']].values.astype(float)
    coh_mat = np.repeat(np.eye((ns), dtype=dtype)[None, :, :], nf, axis=0) # TODO use sparse matrix 
    for ic in range(3):
        Lc = kwargs['l_c'] * [1, 2.7/8.1 , 0.66/8.1 ][ic]
        exp_constant = np.sqrt( (1/ kwargs['u_ref'] * freq)**2 + (0.12 / Lc)**2).astype(dtype)
        i_comp = np.arange(ns)[spat_df.iloc[0, :].values == ic]  # Selecting only u-components
        # loop through number of combinations, nPerChunks at a time to reduce memory impact
        for ii_jj in chunker(itertools.combinations(i_comp, 2), nPerChunks=10000):
            # get indices of point-pairs
            ii = np.array([tup[0] for tup in ii_jj])
            jj = np.array([tup[1] for tup in ii_jj])
            # calculate distances and coherences
            r = np.sqrt((yz[0, ii] - yz[0, jj])**2 + (yz[1, ii] - yz[1, jj])**2)
            coh_values = np.exp(-12 * np.abs(r) * exp_constant)
            # Previous method (same math, different numerics)
            coh_mat[:, ii, jj] = coh_values  # nf x nchunk
            coh_mat[:, jj, ii] = np.conj(coh_values)
    return coh_mat
