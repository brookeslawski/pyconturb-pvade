# -*- coding: utf-8 -*-
"""Functions related to the simulation of turbulence
"""
import itertools

import numpy as np
import pandas as pd

from .coherence import get_coherence
from .helpers import get_iec_sigk, spat_to_pair_df
from .spectra import get_spectrum
from .wind_profiles import get_wsp_profile


def gen_turb(spat_df,
             coh_model='iec', spc_model='kaimal', wsp_profile='iec',
             scale=True, seed=False, **kwargs):
    """Generate turbulence box

    Notes
    -----
    This turbulence box is defined according to the x, y, z coordinate system
    in the HAWC2 coordinate system. In particular, x is directed upwind, z is
    vertical up, and y is lateral to form a right-handed coordinate system.
    """

    # define time vector
    n_t = np.ceil(kwargs['T'] / kwargs['dt'])
    t = np.arange(n_t) * kwargs['dt']

    # create dataframe with magnitudes
    mag_df = get_magnitudes(spat_df, spc_model=spc_model, scale=scale,
                            **kwargs)

    # create dataframe with phases
    pha_df = get_phasors(spat_df,
                         coh_model=coh_model, seed=seed, **kwargs)

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
                   spc_model='kaimal', scale=True, **kwargs):
    """Create dataframe of magnitudes with desired power spectra
    """
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
    n_f = n_t // 2 + 1
    df = 1 / kwargs['T']
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
                coh_model='iec', seed=None, **kwargs):
    """Create realization of phasors with desired coherence

    Notes
    -----
    A phasor is the correlated complex Fourier component that contains the
    phase information, but not the magnitude information. The uncorrelated
    phasors are magnitude 1, but the correlated phasors are not.
    """
    n_f = int(np.ceil(kwargs['T'] / kwargs['dt'])//2 + 1)  # no. of freqs
    freq = np.arange(n_f) / kwargs['T']  # frequency array
    n_s = spat_df.shape[0]

    pair_df = spat_to_pair_df(spat_df)
    coh_df = get_coherence(pair_df, freq, coh_model='iec', **kwargs)

    np.random.seed(seed=seed)  # initialize random number generator
    unc_pha = 2 * np.pi * np.random.rand(n_s, n_f)
    pha_df = pd.DataFrame(np.empty((n_s, n_f)),
                          columns=freq, dtype=complex)

    ii_jj = [(i, j) for (i, j) in itertools.combinations(spat_df.index, 2)]
    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]
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
