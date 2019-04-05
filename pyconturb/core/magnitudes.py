# -*- coding: utf-8 -*-
"""Functions related to definitions of spectral models
"""
import warnings

import numpy as np
import pandas as pd

from .helpers import get_iec_sigk, spc_to_mag


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
    mag_df : pd.DataFrame
        Values of spectral model for specified spatial data and frequency.
        Index is point, column is frequency.
    """
    # pull out con data for easier access later
    con_spat_df, con_turb_df = con_data['con_spat_df'], con_data['con_turb_df']

    # check that all components requested have 1+ constraint given
    if set(spat_df.k) - set(con_spat_df.k):
        raise ValueError('Cannot interpolate component with no data!')

    # also check that interp method is given
    if 'method' not in kwargs.keys():
        raise ValueError('Missing key "method" for data interpolation!')

    # initialize the output magnitude dataframe
    mag_cols = [f'{"uvw"[k]}_p{i}' for (k, i) in zip(spat_df.k, spat_df.p_id)]
    mags_df = pd.DataFrame(index=freq, columns=mag_cols, dtype=float)

    # interpolate by vertical height
    if kwargs['method'] == 'z_interp':

        # calculate the df with the constraining magnitudes, linearly averaging
        # any magnitudes of the same turb component at the same vertical height
        uniq_spat_df = pd.DataFrame(columns=con_spat_df.columns)
        uniq_mag_df = pd.DataFrame(index=freq)
        for k in con_spat_df.k.unique():
            for z in con_spat_df[con_spat_df.k == k].z.unique():
                sub_spat_df = con_spat_df[(con_spat_df.k == k) &
                                          (con_spat_df.z == z)]
                uniq_spat_df = uniq_spat_df.append(sub_spat_df.iloc[0, :])
                sub_pids = [f'{"uvw"[k]}_p{i}' for (k, i) in
                            zip(sub_spat_df.k, sub_spat_df.p_id)]
                sub_turb_df = con_turb_df[sub_pids]
                mag_mean = np.abs(np.fft.rfft(sub_turb_df,
                                              axis=0)).mean(axis=1)
                uniq_mag_df[sub_pids[0]] = mag_mean

        # loop through each component
        for k in spat_df.k.unique():

            # get magnitudes for interpolation
            sub_spat_df = uniq_spat_df[uniq_spat_df.k == k]  # con pts for k
            sub_ptnms = [f'{"uvw"[k]}_p{i}' for (k, i) in
                         zip(sub_spat_df.k, sub_spat_df.p_id)]  # con pt names
            sub_mag_df = uniq_mag_df[sub_ptnms]  # mags for those con pts

            # create dataframe for interpolating
            intp_df = sub_mag_df.T
            intp_df.index = sub_spat_df.z

            # add rows for new values and interpolate them
            sim_zs = spat_df[spat_df.k == k].z
            comb_idx = intp_df.index.union(set(sim_zs))
            intp_df = intp_df.reindex(comb_idx).interpolate(method='values')\
                .fillna(method='bfill')
            res_df = intp_df.loc[sim_zs, :].T

            # assign values to mags_df
            col_names = [f'{"uvw"[k]}_p{i}' for i in spat_df[spat_df.k == k].p_id.values]
            mags_df[col_names] = res_df.values

    else:
        raise ValueError('Method is not defined!')
    mags_df = mags_df / con_turb_df.shape[0]
    return mags_df.values


def get_kaimal_spectrum(spat_df, freq,
                        **kwargs):
    """Get Kaimal spectrum for given locations/components and frequencies
    """
    if 'ed' not in kwargs.keys():
        warnings.warn('No IEC edition specified -- assuming Ed. 3')
        kwargs['ed'] = 3
    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['v_hub', 'i_ref']]):
        raise ValueError('Missing keyword arguments for IEC coherence model')

    comps = spat_df.k.values
    z = spat_df.z.values
    freq = np.asarray(freq).reshape(-1, 1)  # need this to be a col vector
    lambda_1 = 0.7 * z * (z < 60) + 42 * (z >= 60)
    l_k = 8.1 * lambda_1 * (comps == 0) + \
        2.7 * lambda_1 * (comps == 1) + \
        0.66 * lambda_1 * (comps == 2)
    sig_k = get_iec_sigk(spat_df, **kwargs).reshape(1, -1)
    tau = (l_k / kwargs['v_hub']).reshape(1, -1)  # L_k / U

    spc_np = (sig_k**2) * (4 * tau) / \
        np.power(1. + 6 * tau * freq, 5. / 3.)  # Kaimal 1972

    return spc_np


def get_magnitudes(spat_df, con_data=None,
                   spc_model='kaimal', **kwargs):
    # define useful parameters
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
    n_f = n_t // 2 + 1
    df = 1 / kwargs['T']
    freq = np.arange(n_f) * df

    # load magnitudes as desired
    if spc_model == 'kaimal':
        spc_np = get_kaimal_spectrum(spat_df, freq, **kwargs)
        mags_np = spc_to_mag(spc_np, spat_df, df, n_t, **kwargs)

    elif spc_model == 'data':
        mags_np = get_data_magnitudes(spat_df, freq, con_data, **kwargs)

    else:
        raise ValueError(f'No such spc_model "{spc_model}"')

    return mags_np
