# -*- coding: utf-8 -*-
"""Functions related to definitions of spectral models
"""
import warnings

import numpy as np
import pandas as pd

from .helpers import get_iec_sigk


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
        spc_df = get_spectrum(spat_df, freq, spc_model=spc_model, **kwargs)
        mags = np.sqrt(spc_df * df / 2)
        mags.iloc[0, :] = 0.  # set dc component to zero

        if kwargs['scale']:
            sum_magsq = 2 * (mags ** 2).sum(axis=0).values.reshape(1, -1)
            sig_k = get_iec_sigk(spat_df, **kwargs).reshape(1, -1)
            alpha = np.sqrt((n_t - 1) / n_t
                            * (sig_k ** 2) / sum_magsq)  # scaling factor
        else:
            alpha = 1
        mags = alpha * mags

#    elif spc_model == 'data':
#        mags = get_data_magnitudes(spat_df, freq, con_data, **kwargs)

    return mags


def get_spectrum(spat_df, freq, con_data=None,
                 spc_model='kaimal', **kwargs):
    """Power spectrum for turbulent component and spatial location

    Calls spectral-model-specific subfunctions.

    Notes
    -----
    The spatial coordinate system is x positive to the right, z positive up,
    and y positive downwind. The turbulence components are indexed such that u
    is 0, v is 1, and w is 2.

    Parameters
    ----------
    spat_df : pd.DataFrame
        Pandas dataframe with spatial location/turbulence component information
        necessary for spectral calculations. The dataframe must have the
        following columns: k (turbulence component index), x, y, and z (spatial
        coordinates).
    freq : array_like
        Frequency/ies at which to evaluate the spectral model.
    spc_model : str, optional
        Spectral model to use. Default is Kaimal spectrum.
    kwargs : dict, optional
        Keyword arguments for specific spectral model.

    Returns
    -------
    spc_df : pd.DataFrame
        Values of spectral model for specified spatial data and frequency.
        Index is point, column is frequency.
    """

    if spc_model == 'kaimal':  # Kaimal spectral model
        if 'ed' not in kwargs.keys():  # add IEC ed to kwargs if not passed in
            kwargs['ed'] = 3
        spc_df = get_kaimal_spectrum(spat_df, freq, **kwargs)

    elif spc_model == 'data':
        spc_df = get_data_spectrum(spat_df, con_data, **kwargs)

    else:  # unknown coherence model
        raise ValueError(f'Spectral model "{spc_model}" not recognized.')

    return spc_df


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
