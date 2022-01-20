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
from pyconturb.sig_models import iec_sig, data_sig
from pyconturb.spectral_models import kaimal_spectrum, data_spectrum
from pyconturb.wind_profiles import get_wsp_values, power_profile, data_profile
from pyconturb._utils import (combine_spat_con, _spat_rownames, _DEF_KWARGS,
                              clean_turb, check_sims_collocated, get_freq, get_chunk_idcs,
                              message)


def gen_turb(spat_df, T=600, nt=600, con_tc=None, coh_model='iec', coh_file=None,
             wsp_func=None, sig_func=None, spec_func=None,
             interp_data='none', seed=None, nf_chunk=1, dtype=np.float64, verbose=False,
             **kwargs):
    """Generate a turbulence box (constrained or unconstrained).

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have rows `[k, x, y, z]`,
        and each of the `n_sp` columns corresponds to a different spatial location and
        turbine component (u, v or w).
    T : float, optional
        Total length of time to simulate in seconds. Default is 600.
    nt : int, optional
        Number of time steps for generated turbulence. Default is 600.
    con_tc : pyconturb TimeConstraint, optional
        Optional constraining data for the simulation. The TimeConstraint object is built
        into PyConTurb; see documentation for more details.
    coh_model : str, optional
        Spatial coherence model specifier. Default is IEC 61400-1, Ed. 4.
    coh_file : str or pathlib.Path
        Path to file from which to load coherence. Assumed to be an HDF5 file with
        dataset "coherence" containing a 2D coherence array with dimensions 
        ``(n_f, (n_sp + n_c)^2)``. See documentation for more details.
    wsp_func : function, optional
        Function to specify spatial variation of mean wind speed. See details
        in `Mean wind speed profiles` section.
    sig_func : function, optional
        Function to specify spatial variation of turbulence standard deviation.
        See details in `Turbulence standard deviation` section.
    spec_func : function, optional
        Function to specify spatial variation of turbulence power spectrum. See
        details in `Turbulence spectra` section.
    interp_data : str, optional
        Interpolate mean wind speed, standard deviation, and/or power spectra profile
        functions from provided constraint data. Possible options are ``'none'`` (use
        provided/default profile functions), ``'all'`` (interpolate all profile,
        functions), or a list containing and combination of ``'wsp'``, ``'sig'`` and
        ``'spec'`` (interpolate the wind speed, standard deviation and/or power spectra).
        Default is IEC 61400-1 profiles (i.e., no interpolation).
    seed : int, optional
        Optional random seed for turbulence generation. Use the same seed and
        settings to regenerate the same turbulence box.
    nf_chunk : int, optional
        Number of frequencies in a chunk of analysis. Increasing this number may speed
        up computation but may result in more (or too much) memory used. Smaller grids
        may benefit from larger values for ``nf_chunk``. Default is 1.
    verbose : bool, optional
        Print extra information during turbulence generation. Default is False.
    dtype : data type, optional
        Change precision of calculation (np.float32 or np.float64). Will reduce the 
        storage, and might slightly reduce the computational time. Default is np.float64.
    **kwargs
        Optional keyword arguments to be fed into the
        spectral/turbulence/profile/etc. models.

    Returns
    -------
    turb_df : pandas.DataFrame
        Generated turbulence box. Each row corresponds to a time step and each
        column corresponds to a point/component in ``spat_df``.
    """
    message('Beginning turbulence simulation...', verbose)

    # if con_data passed in, throw error
    if 'con_data' in kwargs:
        raise ValueError('The "con_data" option is deprecated and can no longer be used.'
                         ' Please see documentation for updated usage.')
    # if dt passed in, throw deprecation warning
    if 'dt' in kwargs:
        warnings.warn('The "dt" keyword argument is deprecated and will be removed in '
                      + 'future releases. Please use the "nt" argument instead.',
                      DeprecationWarning)
        nt = int(np.ceil(T / kwargs['dt']))  # no. time steps
    # if asked to interpret but no data, throw warning
    if (((interp_data == 'all') or isinstance(interp_data, list)) and (con_tc is None)):
        raise ValueError('If "interp_data" is not "none", constraints must be given!')
    # return None if all simulation points collocated with constraints
    if check_sims_collocated(spat_df, con_tc):
        print('All simulation points collocated with constraints! '
              + 'Nothing to simulate.')
        return None

    dtype_complex = np.complex64 if dtype==np.float32 else np.complex128  # complex dtype

    # add T, nt, con_tc to kwargs
    kwargs = {**_DEF_KWARGS, **kwargs, 'T': T, 'nt': nt, 'con_tc': con_tc}
    wsp_func, sig_func, spec_func = assign_profile_functions(wsp_func, sig_func,
                                                             spec_func, interp_data)

    # assign/create constraining data
    t, freq = get_freq(**kwargs)  # time array
    if con_tc is None:  # create empty constraint data if not passed in
        constrained, n_d = False, 0  # no. of constraints
        con_tc = TimeConstraint(index=_spat_rownames)
    else:
        constrained = True
        n_d = con_tc.shape[1]  # no. of constraints
        if not np.allclose(con_tc.get_time().index.values.astype(float), t):
            raise ValueError('TimeConstraint time does not match requested T, dt!')

    # combine data and sim spat_dfs
    all_spat_df = combine_spat_con(spat_df, con_tc)  # all sim points
    n_s = all_spat_df.shape[1]  # no. of total points to simulate
    n_f = freq.size # no. freqs

    one_point = True if n_s == 1 else False  # only one point, skip coherence

    # get magnitudes of points to simulate. (nf, nsim). con_tc in kwargs.
    # TODO: Change this to similar form as get_coh_mat
    sim_mags = get_magnitudes(all_spat_df.iloc[:, n_d:], spec_func, sig_func, **kwargs)

    if constrained:
        conturb_fft = np.fft.rfft(con_tc.get_time().values, axis=0) / nt  # constr fft
        con_mags = np.abs(conturb_fft)  # mags of constraints
        all_mags = np.concatenate((con_mags, sim_mags), axis=1)  # con and sim
    else:
        all_mags = sim_mags  # just sim
    all_mags = all_mags.astype(dtype, copy=False)

    # get uncorrelated phasors for simulation
    np.random.seed(seed=seed)  # initialize random number generator
    sim_unc_pha = np.exp(1j * 2*np.pi * np.random.rand(n_f, n_s - n_d))
    if not (nt % 2):  # if even time steps, last phase must be 0 or pi for real sig
        sim_unc_pha[-1, :] = np.exp(1j * np.round(np.real(sim_unc_pha[-1, :])) * np.pi)

    # no coherence if one point
    if one_point:
        turb_fft = all_mags * sim_unc_pha

    # if more than one point, correlate everything
    else:
        turb_fft = np.zeros((n_f, n_s), dtype=dtype_complex)
        n_chunks = int(np.ceil(freq.size / nf_chunk))

        # loop through frequencies
        for i_f in range(1, freq.size):
            i_chunk = i_f // nf_chunk  # calculate chunk number
            if (i_f - 1) % nf_chunk == 0:  # genr cohrnc chunk when needed
                message(f'  Processing chunk {i_chunk + 1} / {n_chunks}', verbose)
                
                chunk_idcs = get_chunk_idcs(freq, i_chunk, nf_chunk)
                chunk_coh_mat = get_coh_mat(freq, spat_df, coh_model=coh_model,
                                            chunk_idcs=chunk_idcs, dtype=dtype,
                                            coh_file=coh_file, **kwargs)

            # coherence array for this frequency
            coh_mat = chunk_coh_mat[i_f % nf_chunk]  # ns x ns

            # get cholesky decomposition of coherence matrix, make coherence
            L_mat = coh_mat * all_mags[i_f]

            # if constraints, assign data unc_pha
            if constrained:
                dat_unc_pha = np.linalg.solve(L_mat[:n_d, :n_d], conturb_fft[i_f, :])
            else:
                dat_unc_pha = []
            unc_pha = np.concatenate((dat_unc_pha, sim_unc_pha[i_f, :]))
            cor_pha = L_mat @ unc_pha

            # calculate and save correlated Fourier components
            turb_fft[i_f, :] = cor_pha

        del chunk_coh_mat  # free up memory

    # convert to time domain and pandas dataframe
    turb_arr = np.fft.irfft(turb_fft, axis=0, n=nt) * nt
    turb_arr = turb_arr.astype(dtype, copy=False)
    turb_df = pd.DataFrame(turb_arr, columns=all_spat_df.columns, index=t)

    # return just the desired simulation points
    turb_df = clean_turb(spat_df, all_spat_df, turb_df)

    # add in mean wind speed according to specified profile
    wsp_profile = get_wsp_values(spat_df, wsp_func, **kwargs)
    turb_df[:] += wsp_profile

    message('Turbulence generation complete.', verbose)

    return turb_df


def assign_profile_functions(wsp_func, sig_func, spec_func, interp_data):
    """Assign profile functions based on user inputs"""
    # assign iec profile functions as default if nothing was passed in
    prof_funcs = [wsp_func, sig_func, spec_func]  # given inputs
    iec_funcs = [power_profile, iec_sig,  kaimal_spectrum]  # iec default functions
    prof_funcs = [tup[tup[0] is None] for tup in zip(prof_funcs, iec_funcs)]  # overwrite
    # change interp_data to list format if 'all' or 'none' passed in
    if interp_data == 'none': interp_data = []  # no profs interpolated
    elif interp_data == 'all': interp_data = ['wsp', 'sig', 'spec']  # all profs interp'd
    elif not isinstance(interp_data, list):  # bad input
        raise ValueError('"interp_data" must be either "all", "none", or a list!')
    # assign the interp_data profiles IF no custom function was passed in
    for prof in interp_data:
        if (prof == 'wsp') and (wsp_func is None): prof_funcs[0] = data_profile
        elif (prof == 'sig') and (sig_func is None): prof_funcs[1] = data_sig
        elif (prof == 'spec') and (spec_func is None): prof_funcs[2] = data_spectrum
        else: raise ValueError(f'Unknown profile type "{prof}"!')  # bad input
    return prof_funcs
