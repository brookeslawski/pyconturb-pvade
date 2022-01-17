# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""
import itertools
import os

import h5py
import numpy as np
from scipy.linalg import cholesky

from pyconturb._utils import check_chunk_idcs


_HDF5_DSNAME = 'coherence'  # dataset name for HDF5 coherence files

def calculate_coh_mat(freq, spat_df, coh_model='iec', dtype=np.float64, chunk_idcs=None,
                      **kwargs):
    """Calculate coherence matrix using Cholesky decomposition.

    Parameters
    ----------
    freq : array-like
        [Hz] Full frequency vector for coherence calculations. Option to calculate 
        coherence for a subset using `chunk_idcs` keyword argument. Dimension is
        ``(n_f,)``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have rows `[k, x, y, z]`,
        and each of the `n_sp` columns corresponds to a different spatial location and
        turbine component (u, v or w).
    coh_model : str, optional
        Spatial coherence model specifier. Default is "iec" (IEC 61400-1, Ed. 4).
    dtype : data type, optional
        Change precision of calculation (np.float32 or np.float64). Will reduce the 
        storage, and might slightly reduce the computational time. Default is np.float64.
    chunk_idcs : int or numpy.array
        Indices of `freq` for which coherence should be calculated or loaded. Dimension
        is ``(n_fchunk,)`` if given. Default is None (get all frequencies in `freq`).
    **kwargs
        Keyword arguments to pass into ``get_iec[3d]_cor_mat``.


    Returns
    -------
    coh_mat : numpy.ndarray
        Generated coherence matrix. Dimension is ``(n_fchunk, n_sp, n_sp)``.
    """
    
    # update chunk_idcs if not given
    chunk_idcs = check_chunk_idcs(freq, chunk_idcs)
    
    # throw error if chunk indices not a numpy array
    if type(chunk_idcs) not in [int, np.ndarray]:
        raise ValueError('chunk_idcs must be integer or numpy array!')
    
    # get chunk frequency array
    freq_chunk = np.atleast_1d(freq)[chunk_idcs]

    # get CORRELATION matrix based on coherence model
    if coh_model == 'iec':  # IEC coherence model
        coh_mat = get_iec_cor_mat(freq_chunk, spat_df, dtype=dtype, coh_model=coh_model,
                                  **kwargs)
    elif coh_model == 'iec3d':  # IEC coherence model, but in u, v, and w
        coh_mat = get_iec3d_cor_mat(freq_chunk, spat_df, dtype=dtype, coh_model=coh_model,
                                    **kwargs)
    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized!')

    # get COHERENCE matrix (in-place) via cholesky decomp. In-place saves memory.
    # TODO! EASY parallelization here
    for i in range(coh_mat.shape[0]):
        coh_mat[i] = cholesky(coh_mat[i], check_finite=False, lower=True)

    return coh_mat



def chunker(iterable, nPerChunks):
    """Return list of nPerChunks elements of an iterable"""
    it = iter(iterable)
    while True:
       chunk = list(itertools.islice(it, nPerChunks))
       if not chunk:
           return
       yield chunk


def get_coh_mat(freq, spat_df, coh_model='iec', dtype=np.float64, coh_file=None,
                chunk_idcs=None, **kwargs):
    """Get coherence matrix (either calculate or load) for set of frequencies.
    
    If the `coh_file` option is given, this function calls `load_coh_mat` to load the
    requested matrix from file. If no file name is given, this function instead calls
    `calculate_coh_mat` to calculate the coherence matrix.
    
    Parameters
    ----------
    freq : array-like
        [Hz] Full frequency vector for coherence calculations. Option to calculate 
        coherence for a subset using `chunk_idcs` keyword argument. Dimension is
        ``(n_f,)``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have rows `[k, x, y, z]`,
        and each of the `n_sp` columns corresponds to a different spatial location and
        turbine component (u, v or w).
    coh_model : str, optional
        Spatial coherence model specifier. Default is "iec" (IEC 61400-1, Ed. 4).
    dtype : data type, optional
        Change precision of calculation (np.float32 or np.float64). Will reduce the 
        storage, and might slightly reduce the computational time. Default is np.float64.
    coh_file : str or pathlib.Path
        Path to file from which to load coherence. Assumed to be an HDF5 file with
        dataset "coherence" containing a 2D coherence array with dimensions 
        ``(n_f, n_sp^2)``. Default is None (calculate, don't load from file).
    chunk_idcs : int or numpy.array
        Indices of `freq` for which coherence should be calculated or loaded. Dimension
        is ``(n_fchunk,)`` if given. Default is None (get all frequencies in `freq`).
    **kwargs
        Keyword arguments to pass into ``calculate_coh_mat``.


    Returns
    -------
    coh_mat : numpy.ndarray
        Generated coherence matrix. Dimension is ``(n_fchunk, n_sp, n_sp)``.
    """
    
    # update chunk_idcs if not given
    chunk_idcs = check_chunk_idcs(freq, chunk_idcs)

    # if filename not given, calculate the coherence matrix
    if coh_file is None:
        coh_mat = calculate_coh_mat(freq, spat_df, coh_model=coh_model, dtype=dtype,
                                    chunk_idcs=chunk_idcs, **kwargs)
    
    # if a filename IS given
    else:
        
        # raise error if file not existing
        if not os.path.isfile(coh_file):
            raise ValueError(f'File "{coh_file}" not found! Generate it using `generate_coherence_file`.')
        
        # otherwise, load it from the file
        coh_mat = load_coh_mat(coh_file, freq, chunk_idcs=chunk_idcs)

    return coh_mat


def generate_coherence_file(freq, spat_df, coh_file, coh_model='iec', nf_chunk=1,
                            dtype=np.float64, **kwargs):
    """Calculate a coherence matrix and save it to an HDF5 file for later reuse.

    Parameters
    ----------
    freq : array-like
        [Hz] Full frequency vector for coherence calculations. Option to calculate 
        coherence for a subset using `chunk_idcs` keyword argument. Dimension is
        ``(n_f,)``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have rows `[k, x, y, z]`,
        and each of the `n_sp` columns corresponds to a different spatial location and
        turbine component (u, v or w).
    coh_file : str or pathlib.Path
        Path to file from which to load coherence. Assumed to be an HDF5 file with
        dataset "coherence" containing a 2D coherence array with dimensions 
        ``(n_f, n_sp^2)``.
    coh_model : str, optional
        Spatial coherence model specifier. Default is "iec" (IEC 61400-1, Ed. 4).
    dtype : data type, optional
        Change precision of calculation (np.float32 or np.float64). Will reduce the 
        storage, and might slightly reduce the computational time. Default is np.float64.
    nf_chunk : int, optional
        Number of frequencies in a chunk of analysis. Increasing this number may speed
        up computation but may result in more (or too much) memory used. Smaller grids
        may benefit from larger values for ``nf_chunk``. Default is 1.
    **kwargs
        Keyword arguments to pass into ``calculate_coh_mat``.


    Returns
    -------
    coh_mat : numpy.ndarray
        Generated coherence matrix. Dimension is ``(n_fchunk, n_sp, n_sp)``.
    """
    n_s = spat_df.shape[1]
    n_f = np.size(freq)
    n_chunks = np.ceil(n_f / nf_chunk).astype(int)
    
    # open the file
    with h5py.File(coh_file, 'w') as hf:
        
        # initialize dataset
        dset = hf.create_dataset(_HDF5_DSNAME, (n_f, n_s**2), dtype=dtype)
        
        # for each chunk
        for i_chunk in range(n_chunks):
            
            # get the chunk indices
            chunk_idcs = np.arange(i_chunk*n_chunks, min((i_chunk + 1)*n_chunks, n_f))
            
            # calculate the coherence matrix
            chunk_coh_mat = calculate_coh_mat(freq, spat_df, coh_model=coh_model,
                                              dtype=dtype, chunk_idcs=chunk_idcs,
                                              **kwargs)

            # add coherence to the dataset
            dset[chunk_idcs, :] = chunk_coh_mat.reshape(len(chunk_idcs), n_s**2)
            
    return


def get_iec_cor_mat(freq, spat_df, dtype=np.float64, ed=4, **kwargs):
    """Create IEC 61400-1 Ed. 3/4 correlation matrix for given frequencies. [nf x ns x ns]
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
    cor_mat = np.repeat(np.eye((ns), dtype=dtype)[None, :, :], nf, axis=0) # TODO use sparse matrix 
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
        cor_mat[:, ii, jj] = coh_values  # nf x nchunk
        cor_mat[:, jj, ii] = np.conj(coh_values)
    return cor_mat


def get_iec3d_cor_mat(freq, spat_df, dtype=np.float64, **kwargs):
    """Create IEC-flavor correlation but for all 3 components. Returns nf x ns x ns.
    """
    # preliminaries
    if any([k not in kwargs for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    freq = np.array(freq).reshape(-1, 1)  # nf x 1
    nf, ns = freq.size, spat_df.shape[1]
    # intermediate variables
    yz = spat_df.loc[['y', 'z']].values.astype(float)
    cor_mat = np.repeat(np.eye((ns), dtype=dtype)[None, :, :], nf, axis=0) # TODO use sparse matrix 
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
            cor_mat[:, ii, jj] = coh_values  # nf x nchunk
            cor_mat[:, jj, ii] = np.conj(coh_values)
    return cor_mat



def load_coh_mat(coh_file, freq, chunk_idcs=None):
    """Load all or part of a coherence matrix from an HDF5 file.
    
    Parameters
    ----------
    coh_file : str or pathlib.Path
        Path to file from which to load coherence. Assumed to be an HDF5 file with
        dataset "coherence" containing a 2D coherence array with dimensions 
        ``(n_f, n_sp^2)``.
    freq : array-like
        [Hz] Full frequency vector for coherence calculations. Option to calculate 
        coherence for a subset using `chunk_idcs` keyword argument. Dimension is
        ``(n_f,)``.
    chunk_idcs : int or numpy.array
        Indices of `freq` for which coherence should be calculated or loaded. Dimension
        is ``(n_fchunk,)`` if given. Default is None (get all frequencies in `freq`).
    
    
    Returns
    -------
    coh_mat : numpy.ndarray
        Generated coherence matrix. Dimension is ``(n_fchunk, n_sp, n_sp)``.
    """
    
    # update chunk_idcs if not given
    chunk_idcs = check_chunk_idcs(freq, chunk_idcs)
    
    # load the slice from memory
    with h5py.File(coh_file, 'r') as hf:
        coh_mat_2d = hf[_HDF5_DSNAME][chunk_idcs, :]
    
    ns = np.sqrt(coh_mat_2d.shape[1]).astype(int)
    
    return coh_mat_2d.reshape((-1, ns, ns))
    
