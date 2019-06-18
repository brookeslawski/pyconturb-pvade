# -*- coding: utf-8 -*-
"""Create an unconstrained or constrained turbulence box.

The main function function, ``gen_turb`` can be used both with and without
constraining data. Please see the Examples section in the documentation
to see how to use it.
"""
import warnings

import numpy as np
import pandas as pd

from pyconturb.coherence import get_coh_mat
from pyconturb.core import TimeConstraint
from pyconturb.magnitudes import get_magnitudes
from pyconturb.wind_profiles import get_wsp_values, power_profile, data_profile
from pyconturb.spectral_models import kaimal_spectrum, data_spectrum
from pyconturb.sig_models import iec_sig, data_sig
from pyconturb._utils import combine_spat_con, _spat_rownames, _DEF_KWARGS, clean_turb


def gen_turb(spat_df, T=600, dt=1, con_tc=None, coh_model='iec',
             wsp_func=None, sig_func=None, spec_func=None,
             seed=None, mem_gb=0.10, verbose=False, **kwargs):
    """Generate a turbulence box (constrained or unconstrained).

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have rows `[k, x, y, z]`,
        and each of the `n_sp` columns corresponds to a different spatial location and
        turbine component (u, v or w).
    T : float, optional
        Total length of time to simulate in seconds. Default is 600.
    dt : float, optional
        Time step for generated turbulence in seconds. Default is 1.
    con_tc : pyconturb TimeConstraint, optional
        Optional constraining data for the simulation. The TimeConstraint object is built
        into PyConTurb; see documentation for more details.
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
        column corresponds to a point/component in ``spat_df``.
    """
    if verbose:
        print('Beginning turbulence simulation...')
    if ('con_data' in kwargs) and (con_tc is None):  # don't use con_data please
        warnings.warn('The con_data option is deprecated and will be removed in future' +
                      ' versions. Please see the documentation for how to specify' +
                      ' time constraints.',
                      DeprecationWarning, stack_level=2)
        con_tc = TimeConstraint().from_con_data(kwargs['con_data'])
    elif ('con_data' in kwargs) and (con_tc is not None):  # both passed in
        warnings.warn('Both con_data (deprecated) and con_tc passed in! Using con_tc.',
                      DeprecationWarning, stack_level=2)

    # add T, dt, con_tc to kwargs, assign profile functions (default or data if con)
    kwargs = {**_DEF_KWARGS, **kwargs, 'T': T, 'dt': dt, 'con_tc': con_tc}
    prof_funcs = [[wsp_func, power_profile, data_profile], [sig_func, iec_sig, data_sig],
                  [spec_func, kaimal_spectrum, data_spectrum]]
    for i_func in range(len(prof_funcs)):
        if (prof_funcs[i_func][0] is None) and (con_tc is None):  # no constraint
            prof_funcs[i_func][0] = prof_funcs[i_func][1]
        elif (prof_funcs[i_func][0] is None) and (con_tc is not None):  # constraint
            prof_funcs[i_func][0] = prof_funcs[i_func][2]
    wsp_func, sig_func, spec_func = [prof_funcs[i][0] for i in range(len(prof_funcs))]

    # assign/create constraining data
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))  # no. time steps
    if con_tc is None:  # create empty constraint data if not passed in
        constrained, n_d = False, 0  # no. of constraints
        con_tc = TimeConstraint(index=_spat_rownames)
    else:
        constrained = True
        n_d = con_tc.shape[1]  # no. of constraints
        if con_tc.get_time().shape[0] != n_t:
            print(n_t, con_tc.get_time().shape[0])
            raise ValueError('TimeConstraint time does not match requested T, dt!')

    # combine data and sim spat_dfs
    all_spat_df = combine_spat_con(spat_df, con_tc)  # all sim points
    n_s = all_spat_df.shape[1]  # no. of total points to simulate

    one_point = False
    if n_s == 1:  # only one point, skip coherence
        one_point = True

    # intermediate variables
    n_f = n_t // 2 + 1  # no. freqs
    freq = np.arange(n_f) / kwargs['T']  # frequency array
    t = np.arange(n_t) * kwargs['dt']  # time array

    # get magnitudes of points to simulate. (nf, nsim). con_tc in kwargs.
    sim_mags = get_magnitudes(all_spat_df.iloc[:, n_d:], spec_func, sig_func, **kwargs)

    if constrained:
        conturb_fft = np.fft.rfft(con_tc.get_time().values, axis=0) / n_t  # constr fft
        con_mags = np.abs(conturb_fft)  # mags of constraints
        all_mags = np.concatenate((con_mags, sim_mags), axis=1)  # con and sim
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
        nf_chunk = int(mem_gb * (2 ** 29) / (all_spat_df.shape[1] ** 2))  # no. of freqs in a chunk
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
                dat_unc_pha = np.linalg.solve(cor_mat[:n_d, :n_d], conturb_fft[i_f, :])
            else:
                dat_unc_pha = []
            unc_pha = np.concatenate((dat_unc_pha, sim_unc_pha[i_f, :]))
            cor_pha = cor_mat @ unc_pha

            # calculate and save correlated Fourier components
            turb_fft[i_f, :] = cor_pha

        del all_coh_mat  # free up memory

    # convert to time domain, add mean wind speed profile
    turb_arr = np.fft.irfft(turb_fft, axis=0, n=n_t) * n_t
    turb_df = pd.DataFrame(turb_arr, columns=all_spat_df.columns, index=t)

    # return just the desired simulation points
    turb_df = clean_turb(spat_df, all_spat_df, turb_df)

    # add in mean wind speed according to specified profile
    wsp_profile = get_wsp_values(spat_df, wsp_func, **kwargs)
    turb_df[:] += wsp_profile

    if verbose:
        print('Turbulence generation complete.')

    return turb_df
