# -*- coding: utf-8 -*-
"""Create an unconstrained or constrained turbulence box.

The main function function, ``gen_turb`` can be used both with and without
constraining data. Please see the Examples section in the documentation
to see how to use it.
"""
import numpy as np
import pandas as pd

from pyconturb.coherence import get_coh_mat
from pyconturb.magnitudes import get_magnitudes
from pyconturb.wind_profiles import get_wsp_values, power_profile
from pyconturb.spectral_models import kaimal_spectrum
from pyconturb.sig_models import iec_sig
from pyconturb._utils import combine_spat_df, _spat_colnames, _DEF_KWARGS


def gen_turb(sim_spat_df, T=600, dt=1, con_data=None, coh_model='iec',
             wsp_func=None, sig_func=None, spec_func=None,
             seed=None, mem_gb=0.10, verbose=False, **kwargs):
    """Generate a turbulence box.

    Parameters
    ----------
    sim_spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        `[k, p_id, x, y, z]`, and each of the `n_sp` rows corresponds to a
        different spatial location and turbuine component (u, v or w).
    T : float, optional
        Total length of time to simulate in seconds. Default is 600.
    dt : float, optional
        Time step for generated turbulence in seconds. Default is 1.
    con_data : dict, optional
        Optional constraining data for the simulation. This dictionary must
        have two keyed elements: ``con_spat_df`` and ``con_turb_df``. Both
        are pandas.DataFrames. Please see the Examples for the format of these
        dataframes. Default is none.
    coh_model : str, optional
        Spatial coherence model specifier. Default is IEC 61400-1.
    wsp_func : function, optional
        Function to specify spatial variation of mean wind speed. See details
        in `Mean wind speed profiles` section.
    sig_func : function, optional
        Function to specify spatial variation of turbulence standard deviation.
        See details in `Turbulence standard deviation` section.
    spec_func : function, optional
        Function to specify spatial variation of turbulence power spectrum. See
        details in `Turbulence spectra` section.
    seed : int, optional
        Optional random seed for turbulence generation. Use the same seed and
        settings to regenerate the same turbulence box.
    mem_gb : float, optional
        Size of memory to use when doing the calculations. Increase this number
        to have faster turbulence generation, but if the number becomes too
        large the generation will fail.
    verbose : bool, optional
        Print extra information during turbulence generation. Default is False.
    **kwargs
        Optional keyword arguments to be fed into the
        spectral/turbulence/profile/etc. models.

    Returns
    -------
    turb_df : pandas.DataFrame
        Generated turbulence box. Each row corresponds to a time step and each
        columns corresponds to a point/component in ``sim_spat_df``.
    """
    if verbose:
        print('Beginning turbulence simulation...')

    # assign/create stuff not passed in and add T, dt to kwargs
    kwargs = {**_DEF_KWARGS, **kwargs, 'T': T, 'dt': dt}
    if wsp_func is None:
        wsp_func = power_profile
    if spec_func is None:
        spec_func = kaimal_spectrum
    if sig_func is None:
        sig_func = iec_sig

    # assign/create constraining data
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))  # no. time steps
    if con_data is None:  # create empty constraint data if not passed in
        constrained, n_d = False, 0  # no. of constraints
        con_spat_df, con_turb_df = pd.DataFrame(columns=_spat_colnames), pd.DataFrame()
    else:
        constrained = True
        con_spat_df, con_turb_df = con_data['con_spat_df'], con_data['con_turb_df']
        n_d = con_spat_df.shape[0]  # no. of constraints
        if con_turb_df.shape[0] != n_t:
            raise ValueError('Time values in keyword arguments do not match constraints')

    # combine data and sim spat_dfs
    all_spat_df = combine_spat_df(con_spat_df, sim_spat_df)  # all sim points
    n_s = all_spat_df.shape[0]  # no. of total points to simulate

    one_point = False
    if n_s == 1:  # only one point, can't use indexing
        one_point = True

    # intermediate variables
    n_f = n_t // 2 + 1  # no. freqs
    freq = np.arange(n_f) / kwargs['T']  # frequency array
    t = np.arange(n_t) * kwargs['dt']  # time array

    # get magnitudes of points to simulate. (nf, nsim). con_data in kwargs.
    sim_mags = get_magnitudes(all_spat_df.iloc[n_d:, :], spec_func, sig_func, **kwargs)

    if constrained:
        conturb_fft = np.fft.rfft(con_turb_df.values,
                                  axis=0) / n_t  # constr fft
        con_mags = np.abs(conturb_fft)  # mags of constraints
        all_mags = np.concatenate((con_mags,
                                   sim_mags), axis=1)  # con and sim
    else:
        all_mags = sim_mags  # just sim

    # get uncorrelated phasors for simulation
    np.random.seed(seed=seed)  # initialize random number generator
    sim_unc_pha = np.exp(1j * 2 * np.pi * np.random.rand(n_f, n_s - n_d))

    # no coherence if one point
    if one_point:
        turb_fft = all_mags * sim_unc_pha

    # if more than one point, correlate everything
    else:
        turb_fft = np.zeros((n_f, n_s), dtype=complex)
        nf_chunk = int(mem_gb * (2 ** 29) /
                       (all_spat_df.shape[0] ** 2))  # no. of freqs in a chunk
        if nf_chunk < 1:  # insufficient memory for requested no. of points
            raise MemoryError('Insufficient memory! Consider increasing ' +
                              'the allowable usable memory or using a bigger' +
                              ' machine.')
        n_chunks = int(np.ceil(freq.size / nf_chunk))

        # loop through frequencies
        for i_f in range(1, freq.size):
            i_chunk = i_f // nf_chunk  # calculate chunk number
            if (i_f - 1) % nf_chunk == 0:  # genr cohrnc chunk when needed
                if verbose:
                    print(f'  Processing chunk {i_chunk + 1} / {n_chunks}')
                all_coh_mat = get_coh_mat(freq[i_chunk * nf_chunk:
                                               (i_chunk + 1) * nf_chunk],
                                          all_spat_df, coh_model=coh_model,
                                          **kwargs)

            # assemble "sigma" matrix, which is coh matrix times mag arrays
            coh_mat = all_coh_mat[:, :, i_f % nf_chunk]
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
            turb_fft[i_f, :] = cor_pha

        del all_coh_mat  # free up memory

    # convert to time domain, add mean wind speed profile
    turb_t = np.fft.irfft(turb_fft, axis=0, n=n_t) * n_t

    # inverse fft and transpose to utilize pandas functions easier
    columns = [f'{"uvw"[int(k)]}_p{int(i)}' for (k, i) in
               zip(all_spat_df.k, all_spat_df.p_id)]
    turb_df = pd.DataFrame(turb_t, columns=columns, index=t)

    # return just the desired simulation points
    out_df = pd.DataFrame(index=turb_df.index)
    for i_sim in sim_spat_df.index:
        k, i, x, y, z = sim_spat_df.loc[i_sim, _spat_colnames]
        out_key = f'{"uvw"[int(k)]}_p{int(i)}'
        turb_pid = all_spat_df[(all_spat_df.k == k) &
                               (all_spat_df.x == x) &
                               (all_spat_df.y == y) &
                               (all_spat_df.z == z)].p_id.values[0]
        turb_key = f'{"uvw"[int(k)]}_p{int(turb_pid)}'
        out_df[out_key] = turb_df[turb_key]

    # add in mean wind speed according to specified profile
    wsp_profile = get_wsp_values(sim_spat_df, wsp_func, **kwargs)
    out_df[:] += wsp_profile

    if verbose:
        print('Turbulence generation complete.')

    return out_df
