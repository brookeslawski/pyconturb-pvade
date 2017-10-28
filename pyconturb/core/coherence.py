# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""

import numpy as np
import pandas as pd


def get_coherence(pair_df, freq,
                  coh_model='iec', **kwargs):
    """KSEC-3D coherence function

    Calls coherence-model-specific subfunctions.

    Notes
    -----
    The spatial coordinate system is x positive to the right, z positive up,
    and y positive downwind. The turbulence components are indexed such that u
    is 0, v is 1, and w is 2.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Pandas dataframe with spatial location/turbulence component information
        that is necessary for coherence calculations. The dataframe must have
        the following columns: k1 (turbulence component at point 1), x1, y1,
        z1 (spatial coordinates of point 1), k2 (turbulence component at point
        2), x2, y2, and z2 (spatial coordinates of point 2).
    freq : array_like
        Frequencies at which to evaluate coherence.
    coh_model : str, optional
        Coherence model to use. Default is IEC Ed. 3.
    kwargs : dict
        Keyword arguments for specified coherence model.

    Returns
    -------
    coh_df : pd.DataFrame
        Values of coherence model for specified spatial data and frequency.
    """

    if coh_model == 'iec':  # IEC coherence model
        if 'ed' not in kwargs.keys():  # add IEC ed to kwargs if not passed in
            kwargs['ed'] = 3
        coh_df = get_iec_coherence(pair_df, freq, **kwargs)

    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized.')

    return coh_df


def get_iec_coherence(pair_df, freq,
                      **kwargs):
    """Exponential coherence specified in IEC 61400-1

    Notes
    -----
    The spatial coordinate system is x positive to the right, z positive up,
    and y positive downwind.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Pandas dataframe with spatial location/turbulence component information
        that is necessary for coherence calculations. The dataframe must have
        the following columns: k1 (turbulence component at point 1), x1, y1,
        z1 (spatial coordinates of point 1), k2 (turbulence component at point
        2), x2, y2, and z2 (spatial coordinates of point 2).
    freq : array_like
        Frequencies for which to evaluate coherence.
    kwargs : dictionary
        Other variables specific to this coherence model.

    Returns
    -------
    coh_df : pd.DataFrame
        Values of coherence model for specified spatial data and frequency.
        Index is the pair of spatial components and columns are frequency.
    """

    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['v_hub', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')

    freq = np.asarray(freq).reshape(-1, 1)  # need this to be a col vector
    coh_df = pd.DataFrame(0, index=freq.reshape(-1),
                          columns=range(pair_df.shape[0]))  # init as zeros

    r = np.sqrt((pair_df.y1 - pair_df.y2)**2 +
                (pair_df.z1 - pair_df.z2)**2).values.reshape(1, -1)
    mask = (pair_df.k1 == 'vxt') & (pair_df.k2 == 'vxt')
    coh_df.loc[:, mask] = np.exp(-12 *
                                 np.sqrt((r / kwargs['v_hub'] * freq)**2 +
                                         (0.12 * r /
                                          kwargs['l_c'])**2))[:, mask]

    return coh_df
