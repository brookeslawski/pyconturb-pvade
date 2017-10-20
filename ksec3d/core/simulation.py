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


#def gen_unc_turb(spat_df,
#                 coh_model='iec', spc_model='kaimal', wsp_model='iec',
#                 scale=True, seed=None, **kwargs):
#    """Generate unconstrained turbulence box
#
#    Notes
#    -----
#    This turbulence box is defined according to the x, y, z coordinate system
#    in the HAWC2 coordinate system. In particular, x is directed upwind, z is
#    vertical up, and y is lateral to form a right-handed coordinate system.
#    """
#
#    # define time vector
#    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
#    t = np.arange(n_t) * kwargs['dt']
#
#    # create dataframe with magnitudes
#    mag_df = get_unc_magnitudes(spat_df, spc_model=spc_model, scale=scale,
#                                **kwargs)
#
#    # create dataframe with phases
#    pha_df = get_unc_phasors(spat_df,
#                             coh_model=coh_model, seed=seed, **kwargs)
#
#    # multiply dataframes together
#    turb_fft = pd.DataFrame(mag_df.values * pha_df.values,
#                            columns=mag_df.columns,
#                            index=pha_df.index)
#
#    # convert to time domain, add mean wind speed profile
#    wsp_profile = get_wsp_profile(spat_df, wsp_model=wsp_model, **kwargs)
#    turb_t = np.fft.irfft(turb_fft, axis=0, n=n_t) * n_t + wsp_profile
#
#    # inverse fft and transpose to utilize pandas functions easier
#    columns = (spat_df.k + '_' + spat_df.p_id).values
#    turb_df = pd.DataFrame(turb_t,
#                           columns=columns,
#                           index=t)
#
#    return turb_df


#def gen_con_turb(data_spat_df, sim_spat_df, conturb_df,
#                 coh_model='iec', spc_model='kaimal', wsp_model='iec',
#                 scale=True, seed=None, **kwargs):
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


#def get_con_magnitudes(all_spat_df, n_d,
#                       spc_model='kaimal', scale=True, **kwargs):
#    """Create dataframe of constrained magnitudes with desired power spectra
#    """
#    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
#    n_f = n_t // 2 + 1
#    df = 1 / kwargs['T']
#    freq = np.arange(n_f) * df
#    spc_df = get_spectrum(all_spat_df, freq, spc_model=spc_model, **kwargs)
#    mags = np.sqrt(spc_df * df / 2)
#    mags.iloc[:, 0] = 0.  # set dc component to zero
#
#    if scale:
#        sum_magsq = 2 * (mags ** 2).sum(axis=0).values.reshape(1, -1)
#        sig_k = get_iec_sigk(all_spat_df, **kwargs).reshape(1, -1)
#        alpha = np.sqrt((n_t - 1) / n_t
#                        * (sig_k ** 2) / sum_magsq)  # scaling factor
#    else:
#        alpha = 1
#    scal_mags = alpha * mags  # scale magnitues
#
#    # set rows corresponding to data to unity (phasors have mag info there)
#    scal_mags.iloc[:n_d, :] = 1
#
#    return scal_mags
#

#def get_unc_phasors(spat_df,
#                    coh_model='iec', seed=None, **kwargs):
#    """Create realization of unconstrained phasors with desired coherence
#
#    Notes
#    -----
#    A phasor is the correlated complex Fourier component that contains the
#    phase information, but not the magnitude information. The uncorrelated
#    phasors are magnitude 1, but the correlated phasors are not.
#    """
#    n_f = int(np.ceil(kwargs['T'] / kwargs['dt'])//2 + 1)  # no. of freqs
#    freq = np.arange(n_f) / kwargs['T']  # frequency array
#    n_s = spat_df.shape[0]
#
#    np.random.seed(seed=seed)  # initialize random number generator
#    unc_pha = np.exp(1j * 2 * np.pi * np.random.rand(n_f, n_s))
#    pha_df = pd.DataFrame(np.empty((n_f, n_s)),
#                          index=freq, dtype=complex)
#
#    pair_df = spat_to_pair_df(spat_df)
#    if pair_df.size == 0:
#        pha_df.loc[:] = unc_pha
#        return pha_df
#    coh_df = get_coherence(pair_df, freq, coh_model='iec', **kwargs)
#
#    ii_jj = [(i, j) for (i, j) in itertools.combinations(spat_df.index, 2)]
#    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]
#    for i_f in range(freq.size):
#        coh_mat = np.ones((n_s, n_s), dtype=complex)
#        coh_mat[ii, jj] = coh_df.iloc[i_f, :].values
#        coh_mat[jj, ii] = np.conj(coh_df.iloc[i_f, :].values)
#        cor_mat = np.linalg.cholesky(coh_mat)
#        pha_df.iloc[i_f, :] = cor_mat @ unc_pha[i_f, :]
#
#    return pha_df
#
#
#def get_con_phasors(data_spat_df, sim_spat_df, conturb_df,
#                    coh_model='iec', seed=None, **kwargs):
#    """Create realization of constrained phasors with desired coherence
#    """
#    # combine dat_spat_df and sim_spat_df to single dataframe
#    all_spat_df = combine_spat_df(data_spat_df, sim_spat_df)
#
#    # define useful parameters
#    n_f = int(np.ceil(kwargs['T'] / kwargs['dt'])//2 + 1)  # no. of freqs
#    freq = np.arange(n_f) / kwargs['T']  # frequency array
#    n_s = all_spat_df.shape[0]  # total number of points
#    n_d = data_spat_df.shape[0]  # number of data points
#
#    # get overall coherence matrix
#    pair_df = spat_to_pair_df(all_spat_df)
#    coh_df = get_coherence(pair_df, freq, coh_model='iec', **kwargs)
#
#    np.random.seed(seed=seed)  # initialize random number generator
#
#    conturb_fft = np.fft.rfft(conturb_df.values, axis=0)  # get fft comps
#
#    # get uncorrelated phasors
#    unc_pha = np.full((n_s, n_f), np.nan)
#    unc_pha[n_d:, :] = np.exp(1j * 2 * np.pi *
#                              np.random.rand(n_s - n_d,
#                                             n_f))  # init sim phasrs to rand
#
#    # assign phasors
#    pha_df = pd.DataFrame(np.empty((n_s, n_f)),
#                          columns=freq, dtype=complex)
#    ii_jj = [(i, j) for (i, j) in itertools.combinations(all_spat_df.index, 2)]
#    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]
#    for i_f in range(freq.size):
#        # get cholesky decomposition of coherence matrix
#        coh_mat = np.ones((n_s, n_s), dtype=complex)
#        coh_mat[ii, jj] = coh_df.iloc[:, i_f].values
#        coh_mat[jj, ii] = np.conj(coh_df.iloc[:, i_f].values)
#        cor_mat = np.linalg.cholesky(coh_mat)
#
#        # assign phasors for data
#        unc_pha[:n_d, i_f] = np.linalg.solve(cor_mat[:n_d, :n_d],
#                                             conturb_fft[i_f, :])
#        pha_df.iloc[:, i_f] = cor_mat @  unc_pha[:, i_f]
#
#    return pha_df
