# -*- coding: utf-8 -*-
"""Test functions in simulation.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from ksec3d.core.simulation import get_unc_phasors, gen_unc_turb,\
                                    get_unc_magnitudes, get_con_magnitudes
from ksec3d.core.helpers import gen_spat_grid


def test_get_phases():
    """Check the value for get_phases
    """
    # given
    spat_df = pd.DataFrame([['vxt', 'p_0', 0, 0, 50],
                            ['vxt', 'p_0', 0, 0, 51]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2,
              'T': 8, 'dt': 4}
    coh_model, seed = 'iec', None
    coh_theory, i_f = 0.8606565849, 1
    rtol = 0.05  # stochastic tolerance
    n_real = 100  # no. realizations for ensemble averaging

    # when
    coh = 0
    den1 = 0
    den2 = 0
    for i_real in range(n_real):
        phasors = get_unc_phasors(spat_df,
                                  coh_model=coh_model, seed=seed,
                                  **kwargs)
        Xi, Xj = phasors.iloc[i_f, 0], phasors.iloc[i_f, 1]
        coh += Xi * np.conj(Xj)
        den1 += Xi * np.conj(Xi)
        den2 += Xj * np.conj(Xj)
    corr = np.real(coh / np.sqrt(den1 * den2))

    # then
    np.testing.assert_allclose(corr, coh_theory, rtol=rtol)


def test_iec_mags_sum():
    """test that the iec magnitudes sum to the right value (or close to)
    """
    # given
    y = [0]  # x-components of turbulent grid
    z = [70]  # z-components of turbulent grid
    spat_df = gen_spat_grid(y, z)
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': z,
              'T': 300, 'dt': 1}
    spc_model = 'kaimal'
    var_theo = np.array([1.834, 1.4672, 0.917]) ** 2

    # when
    mags_ksec = get_unc_magnitudes(spat_df, spc_model=spc_model, scale=True,
                                   **kwargs)
    var_ksec = 2 * (mags_ksec.values ** 2).sum(axis=0)

    # then
    np.testing.assert_allclose(var_ksec, var_theo, rtol=0.01)


def test_iec_turb_mn_std_dev():
    """test that iec turbulence has correct mean and std deviation
    """
    # given
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': 70,
              'T': 300, 'dt': 1}
    coh_model, spc_model = 'iec', 'kaimal'
    sig_theo = np.array([1.834, 1.4672, 0.917, 1.834, 1.4672, 0.917])
    u_theo = np.array([-10, 0, 0, -10.27066087, 0, 0])

    # when
    turb_df = gen_unc_turb(spat_df, coh_model=coh_model, spc_model=spc_model,
                           scale=True, **kwargs)

    # then
    np.testing.assert_allclose(sig_theo, turb_df.std(axis=0),
                               atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, turb_df.mean(axis=0),
                               atol=0.01)

#
#def test_get_con_mags():
#    """Verify that the correct magnitudes are fetched for the constrained case
#    """
#    # given
#    data_spat_df = pd.DataFrame([['vxt', 'p0', 0.0, 0.0, 70.0]],
#                                columns=['k', 'p_id', 'x', 'y', 'z'])
#    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': 70,
#              'T': 300, 'dt': 1}
#    spc_model, scale = 'kaimal', False
#    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
#    y, z = 0, [75, 85]
#    sim_spat_df = gen_spat_grid(y, z)
#    mag_theo = [1 / n_t, 0.5667309913]  # mag_dat(df), mag_u(df)
#    i_f, imag_theo = 1, [0, 1]  # freq index, dat and u index
#
#    # when
#    con_mags = get_con_magnitudes(data_spat_df, sim_spat_df,
#                                  spc_model=spc_model, scale=scale, **kwargs)
#    mag_sim = con_mags.iloc[i_f, imag_theo]
#
#    # then
#    np.testing.assert_allclose(mag_sim, mag_theo)
