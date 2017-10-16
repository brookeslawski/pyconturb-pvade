# -*- coding: utf-8 -*-
"""Functions related to the simulation of turbulence
"""
import itertools

import numpy as np
import pandas as pd

from .coherence import get_coherence
from .helpers import get_iec_sigk
from .spectra import get_spectrum
from .wind_profiles import get_wsp_profile


def gen_turb(spat_df,
             coh_model='iec', spc_model='kaimal', wsp_profile='iec',
             T=600, dt=0.1, scale=True,
             seed=False, **kwargs):
    """Generate turbulence box
    """

    # define time vector
    n_t = np.ceil(T / dt)
    t = np.arange(n_t) * dt

    # create dataframe with magnitudes
    mag_df = get_magnitudes(spat_df, spc_model=spc_model, T=T, dt=dt,
                            scale=scale, **kwargs)

    # create dataframe with phases
    pha_df = get_phasors(spat_df,
                         coh_model='iec', T=T, dt=dt,
                         seed=seed,
                         **kwargs)

    # multiply dataframes together
    turb_fft = pd.DataFrame(mag_df.values * pha_df.values,
                            columns=mag_df.columns,
                            index=pha_df.index)

    # convert to time domain, add mean wind speed profile
    wsp_profile = get_wsp_profile(spat_df,
                                  wsp_model=wsp_profile, **kwargs)
    turb_t = np.fft.irfft(turb_fft, axis=1).T * n_t + wsp_profile

    # inverse fft and transpose to utilize pandas functions easier
    columns = (spat_df.k + '_' + spat_df.p_id).values
    turb_df = pd.DataFrame(turb_t,
                           columns=columns,
                           index=t)

    return turb_df


def get_magnitudes(spat_df,
                   spc_model='kaimal', T=600, dt=0.1, scale=True,
                   **kwargs):
    """Create dataframe of magnitudes with desired power spectra
    """

    n_t = int(np.ceil(T / dt))
    n_f = n_t // 2 + 1
    df = 1 / T
    freq = np.arange(n_f) * df
    spc_df = get_spectrum(spat_df, freq, spc_model=spc_model, **kwargs)
    mags = np.sqrt(spc_df * df / 2)
    mags.iloc[:, 0] = 0.  # set dc component to zero

    if scale:
        sum_magsq = 2 * (mags ** 2).sum(axis=1).values.reshape(-1, 1)
        sig_k = get_iec_sigk(spat_df, **kwargs).reshape(-1, 1)
        alpha = np.sqrt((n_t - 1) / n_t
                        * (sig_k ** 2) / sum_magsq)  # scaling factor
    else:
        alpha = 1

    return alpha * mags


def get_phasors(spat_df,
                coh_model='iec', T=600, dt=0.1, seed=None,
                **kwargs):
    """Create realization of phasors with desired coherence

    Notes
    -----
    A phasor is the correlated complex Fourier component that contains the
    phase information, but not the magnitude information. The uncorrelated
    phasors are magnitude 1, but the correlated phasors are not.
    """

    n_f = int(np.ceil(T / dt)//2 + 1)  # no. of frequencies
    freq = np.arange(n_f) / T  # frequency array
    n_s = spat_df.shape[0]  # no. of spatial points

    n_pairs = int(np.math.factorial(n_s) / 2 /
                  np.math.factorial(n_s - 2))  # no. of combos
    pair_df = pd.DataFrame(np.empty((n_pairs, 8)),
                           columns=['k1', 'x1', 'y1', 'z1', 'k2', 'x2',
                                    'y2', 'z2'])  # df input to coherence fcn

    i_df = 0  # initialize counter
    ii, jj = [], []  # use these index vectors later during cholesky decomp
    for (i, j) in itertools.combinations(spat_df.index, 2):
        pair_df.loc[i_df, ['k1', 'x1', 'y1', 'z1']] = \
            spat_df.loc[i, ['k', 'x', 'y', 'z']].values
        pair_df.loc[i_df, ['k2', 'x2', 'y2', 'z2']] = \
            spat_df.loc[j, ['k', 'x', 'y', 'z']].values
        i_df += 1
        ii.append(i)  # save index
        jj.append(j)  # save index
    coh_df = get_coherence(pair_df, freq, coh_model='iec', **kwargs)

    np.random.seed(seed=seed)  # initialize random number generator
    unc_pha = 2 * np.pi * np.random.rand(n_s, n_f)
    pha_df = pd.DataFrame(np.empty((n_s, n_f)),
                          columns=freq, dtype=complex)

    for i_f in range(freq.size):
        pha_df.iloc[:, i_f] = correlate_phasors(i_f, coh_df, unc_pha,
                                                n_s, ii, jj)

    return pha_df


def correlate_phasors(i_f, coh_df, unc_pha, n_s, ii, jj):
    """Correlate phasors
    """
    coh_mat = np.ones((n_s, n_s), dtype=complex)
    coh_mat[ii, jj] = coh_df.iloc[:, i_f].values
    coh_mat[jj, ii] = np.conj(coh_df.iloc[:, i_f].values)
    cor_mat = np.linalg.cholesky(coh_mat)
    cor_pha = np.dot(cor_mat, np.exp(1j * unc_pha[:, i_f]))
    return cor_pha
