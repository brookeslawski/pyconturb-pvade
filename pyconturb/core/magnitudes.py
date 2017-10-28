# -*- coding: utf-8 -*-
"""Functions related to definitions of spectral models
"""
import warnings

import numpy as np
import pandas as pd

from .helpers import get_iec_sigk, spc_to_mag, complex_interp


def get_data_magnitudes(spat_df, freq, con_data, **kwargs):
    """Get fft magnitudes from data

    Parameters
    ----------
    spat_df : pd.DataFrame
        Pandas dataframe with spatial location/turbulence component information
        that is necessary for coherence calculations. The dataframe must have
        the following columns: k (turbulence component index), x, y, and z
        (spatial coordinates).
    con_data : dict
        Dictionary with con_spat_df and con_turb_df.
    kwargs : dictionary
        Other variables specific to this spectral model.

    Returns
    -------
    spc_df : pd.DataFrame
        Values of spectral model for specified spatial data and frequency.
        Index is point, column is frequency.
    """
    con_spat_df, con_turb_df = con_data['con_spat_df'], con_data['con_turb_df']
    mags_df = pd.DataFrame(index=freq, columns=spat_df.k + '_' + spat_df.p_id)
    con_fft_df = pd.DataFrame(np.fft.rfft(con_turb_df, axis=0),
                              index=freq,
                              columns=con_spat_df.k + '_' +
                              con_spat_df.p_id)
    if kwargs['method'] == 'z_interp':
        for k in spat_df.k.unique():
            con_pids = con_spat_df[con_spat_df.k == k].p_id
            new_pids = spat_df[spat_df.k == k].p_id
            con_zs = con_spat_df[con_spat_df.k == k].z
            intp_df = con_fft_df[k + '_' + con_pids].T
            intp_df.index = con_zs
            new_zs = pd.Index(spat_df[spat_df.k == k].z, dtype=float)
            new_idx = intp_df.index.union(new_zs)
            intp_res = complex_interp(intp_df.reindex(new_idx),
                                      method='values').loc[new_zs]
            mags_df[k + '_' + new_pids] = intp_res.T.values
    else:
        raise ValueError('Method is not defined!')
    return mags_df


def get_kaimal_spectrum(spat_df, freq,
                        **kwargs):
    """One-sided, continuous Kaimal spectrum

    Parameters
    ----------
    spat_df : pd.DataFrame
        Pandas dataframe with spatial location/turbulence component information
        that is necessary for coherence calculations. The dataframe must have
        the following columns: k (turbulence component index), x, y, and z
        (spatial coordinates).
    freq : array_like
        Frequencies for which to evaluate coherence.
    kwargs : dictionary
        Other variables specific to this spectral model.

    Returns
    -------
    spc_df : pd.DataFrame
        Values of spectral model for specified spatial data and frequency.
        Index is point, column is frequency.
    """

    if 'ed' not in kwargs.keys():
        warnings.warn('No IEC edition specified -- assuming Ed. 3')
        kwargs['ed'] = 3
    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['v_hub', 'i_ref']]):
        raise ValueError('Missing keyword arguments for IEC coherence model')

    freq = np.asarray(freq).reshape(1, -1)  # need this to be a row vector
    spc_df = pd.DataFrame(0, index=np.arange(spat_df.shape[0]),
                          columns=freq.reshape(-1))  # initialize to zeros

    # assign/calculate intermediate variables
    spat_df = spat_df.copy()
    spat_df['lambda_1'] = 0.7 * spat_df.mask(spat_df.z > 60, other=0).z + \
        42 * np.sign(spat_df.mask(spat_df.z <= 60, other=0).z)
    l_k = 8.1 * spat_df.mask(spat_df.k != 'vxt', other=0).lambda_1 + \
        2.7 * spat_df.mask(spat_df.k != 'vyt', other=0).lambda_1 + \
        0.66 * spat_df.mask(spat_df.k != 'vzt', other=0).lambda_1  # lngth scl
    sig_k = get_iec_sigk(spat_df, **kwargs).reshape(-1, 1)
    tau = (l_k / kwargs['v_hub']).values.reshape(-1, 1)  # L_k / U

    spc_df[:] = (sig_k**2) * (4 * tau) / \
        np.power(1. + 6 * tau * freq, 5. / 3.)  # Kaimal 1972
    spc_df = spc_df.T  # put frequency along rows

    return spc_df


def get_magnitudes(spat_df, con_data=None,
                   spc_model='kaimal', **kwargs):
    """Create dataframe of unconstrained magnitudes with desired power spectra
    """
    # define useful parameters
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
    n_f = n_t // 2 + 1
    df = 1 / kwargs['T']
    freq = np.arange(n_f) * df

    # load magnitudes as desired
    if spc_model == 'kaimal':
        spc_df = get_kaimal_spectrum(spat_df, freq, **kwargs)
        mags_df = spc_to_mag(spc_df, spat_df, df, n_t, **kwargs)

    elif spc_model == 'data':
        mags_df = get_data_magnitudes(spat_df, freq, con_data, **kwargs)

    else:
        raise ValueError(f'No such spc_model "{spc_model}"')

    return mags_df
