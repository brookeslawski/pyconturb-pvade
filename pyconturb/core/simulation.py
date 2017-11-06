# -*- coding: utf-8 -*-
"""Functions related to the simulation of turbulence
"""
import itertools

import numpy as np
import pandas as pd

from .coherence import get_coherence
from .helpers import spat_to_pair_df, combine_spat_df
from .magnitudes import get_magnitudes
from .wind_profiles import get_wsp_profile


def gen_turb(sim_spat_df, con_data=None,
             coh_model='iec', spc_model='kaimal', wsp_model='iec',
             seed=None, **kwargs):
    """Generate constrained turbulence box

    Notes
    -----
    This turbulence box is defined according to the x, y, z coordinate system
    in the HAWC2 coordinate system. In particular, x is directed upwind, z is
    vertical up, and y is lateral to form a right-handed coordinate system.
    """
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))  # no. time steps
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
        if con_turb_df.shape[0] != n_t:
            raise ValueError('Time values in keyword arguments do not ' +
                             'match constraints')

    # combine data and sim spat_dfs
    all_spat_df = combine_spat_df(con_spat_df, sim_spat_df)  # all sim points
    n_s = all_spat_df.shape[0]  # no. of total points to simulate

    one_point = False
    if n_s == 1:  # only one point
        one_point = True

    # intermediate variables
    n_f = n_t // 2 + 1  # no. freqs
    freq = np.arange(n_f) / kwargs['T']  # frequency array
    t = np.arange(n_t) * kwargs['dt']  # time array
    pair_df = spat_to_pair_df(all_spat_df)  # pairwise info for coherence
    ii_jj = [(i, j) for (i, j) in
             itertools.combinations(all_spat_df.index, 2)]  # pairwise indices
    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]

    # get magnitudes of constraints and from theory
    conturb_fft = np.fft.rfft(con_turb_df.values, axis=0) / n_t  # constr fft
    sim_mags = get_magnitudes(all_spat_df.iloc[n_d:, :],
                              con_data=con_data,
                              spc_model=spc_model,
                              **kwargs)  # mags of sim points
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

    # convert to time domain, add mean wind speed profile
    turb_t = np.fft.irfft(turb_fft, axis=0, n=n_t) * n_t

    # inverse fft and transpose to utilize pandas functions easier
    columns = (all_spat_df.k + '_' + all_spat_df.p_id).values
    turb_df = pd.DataFrame(turb_t,
                           columns=columns,
                           index=t)

    # return either all the points or just the desired simulation points
    out_df = pd.DataFrame(index=turb_df.index)
    for i_sim in sim_spat_df.index:
        k, p_id, x, y, z = sim_spat_df.loc[i_sim,
                                           ['k', 'p_id', 'x', 'y', 'z']]
        out_key = f'{k}_{p_id}'
        turb_pid = all_spat_df[(all_spat_df.k == k) &
                               (all_spat_df.x == x) &
                               (all_spat_df.y == y) &
                               (all_spat_df.z == z)].p_id.values[0]
        turb_key = f'{k}_{turb_pid}'
        out_df[out_key] = turb_df[turb_key]

    # add in mean wind speed according to specified profile
    wsp_profile = get_wsp_profile(sim_spat_df,
                                  wsp_model=wsp_model, **kwargs)
    out_df[:] += wsp_profile

    return out_df
