# -*- coding: utf-8 -*-
"""Functions related to the simulation of turbulence
"""
import itertools

import numpy as np
import pandas as pd

from .coherence import get_coherence
from .helpers import get_iec_sigk, spat_to_pair_df, combine_spat_df
from .spectra import get_spectrum
from .wind_profiles import get_wsp_profile


def gen_turb(sim_spat_df, con_data=None,
             coh_model='iec', spc_model='kaimal', wsp_model='iec',
             scale=True, seed=None, **kwargs):
    """Generate constrained turbulence box

    Notes
    -----
    This turbulence box is defined according to the x, y, z coordinate system
    in the HAWC2 coordinate system. In particular, x is directed upwind, z is
    vertical up, and y is lateral to form a right-handed coordinate system.
    """
    # create empty constraint data if not passed in
    if con_data is None:
        constrained = False
        con_spat_df = pd.DataFrame(np.empty((1, 0)))
        con_turb_df = pd.DataFrame(np.empty((1, 0)))
        n_d = 0  # no. of constraints

    else:
        constrained = True
        con_spat_df = con_data['con_spat_df']
        con_turb_df = con_data['con_turb_df']
        n_d = con_spat_df.shape[0]  # no. of constraints

    # combine data and sim spat_dfs
    all_spat_df = combine_spat_df(con_spat_df, sim_spat_df)  # all sim points
    n_s = all_spat_df.shape[0]  # no. of total points to simulate

    one_point = False
    if n_s == 1:  # only one point
        one_point = True

    # intermediate variables
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))  # no. time steps
    n_f = n_t // 2 + 1  # no. freqs
    freq = np.arange(n_f) / kwargs['T']  # frequency array
    t = np.arange(n_t) * kwargs['dt']  # time array
    pair_df = spat_to_pair_df(all_spat_df)  # pairwise info for coherence
    ii_jj = [(i, j) for (i, j) in
             itertools.combinations(all_spat_df.index, 2)]  # pairwise indices
    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]

    # get magnitudes of constraints and from theory
    conturb_fft = np.fft.rfft(con_turb_df.values, axis=0) / n_t  # constr fft
    sim_mags = get_magnitudes(sim_spat_df, spc_model=spc_model,
                              scale=scale, **kwargs)  # mags of sim points
    if constrained:
        con_mags = np.abs(conturb_fft)  # mags of constraints
        all_mags = np.concatenate((con_mags,
                                   sim_mags.values), axis=1)  # con and sim
    else:
        all_mags = sim_mags.values  # just sim

    # get coherences for pairs of constraint and simlation points
    if not one_point:
        coh_df = get_coherence(pair_df, freq, coh_model=coh_model, **kwargs)

    # get uncorrelated phasors for simulation
    np.random.seed(seed=seed)  # initialize random number generator
    sim_unc_pha = np.exp(1j * 2 * np.pi * np.random.rand(n_f, n_s - n_d))

    # initialize turbulence fft
    turb_fft = pd.DataFrame(np.zeros((n_f, n_s)),
                            index=freq, columns=range(n_s),
                            dtype=complex)

    # loop through frequencies
    for i_f in range(1, freq.size):

        # no coherence if one point
        if one_point:
            cor_pha = all_mags[i_f, :] * sim_unc_pha[i_f, :]

        # otherwise
        else:

            # assemble "sigma" matrix, which is coh matrix times mag arrays
            coh_mat = np.ones((n_s, n_s), dtype=complex)
            coh_mat[ii, jj] = coh_df.iloc[i_f, :].values
            coh_mat[jj, ii] = np.conj(coh_df.iloc[i_f, :].values)
            sigma = np.einsum('i,j->ij', all_mags[i_f, :],
                              all_mags[i_f, :]) * coh_mat

            # get cholesky decomposition of sigma matrix
            cor_mat = np.linalg.cholesky(sigma)

            # if constraints, assign data unc_pha
            if constrained:
                dat_unc_pha = np.linalg.solve(cor_mat[:n_d, :n_d],
                                              conturb_fft[i_f, :])
            else:
                dat_unc_pha = []
            unc_pha = np.concatenate((dat_unc_pha,
                                      sim_unc_pha[i_f, :]))
            cor_pha = cor_mat @ unc_pha

        # calculate and save correlated Fourier components
        turb_fft.iloc[i_f, :] = cor_pha

    # add back in zero-frequency components
    turb_fft.iloc[0, :n_d] = conturb_fft[0, :]

    # convert to time domain, add mean wind speed profile
    turb_t = np.fft.irfft(turb_fft, axis=0, n=n_t) * n_t
    wsp_profile = get_wsp_profile(sim_spat_df, wsp_model=wsp_model, **kwargs)
    turb_t[:, n_d:] += wsp_profile

    # inverse fft and transpose to utilize pandas functions easier
    columns = (all_spat_df.k + '_' + all_spat_df.p_id).values
    turb_df = pd.DataFrame(turb_t,
                           columns=columns,
                           index=t)

    return turb_df


def get_magnitudes(spat_df,
                   spc_model='kaimal', scale=True, **kwargs):
    """Create dataframe of unconstrained magnitudes with desired power spectra
    """
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
    n_f = n_t // 2 + 1
    df = 1 / kwargs['T']
    freq = np.arange(n_f) * df
    spc_df = get_spectrum(spat_df, freq, spc_model=spc_model, **kwargs)
    mags = np.sqrt(spc_df * df / 2)
    mags.iloc[0, :] = 0.  # set dc component to zero

    if scale:
        sum_magsq = 2 * (mags ** 2).sum(axis=0).values.reshape(1, -1)
        sig_k = get_iec_sigk(spat_df, **kwargs).reshape(1, -1)
        alpha = np.sqrt((n_t - 1) / n_t
                        * (sig_k ** 2) / sum_magsq)  # scaling factor
    else:
        alpha = 1

    return alpha * mags
