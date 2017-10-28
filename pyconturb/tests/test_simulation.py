# -*- coding: utf-8 -*-
"""Test functions in simulation.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb.core.simulation import gen_turb, get_magnitudes
from pyconturb.core.helpers import gen_spat_grid


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
    mags_ksec = get_magnitudes(spat_df, spc_model=spc_model, scale=True,
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
    turb_df = gen_turb(spat_df, coh_model=coh_model, spc_model=spc_model,
                       scale=True, **kwargs)

    # then
    np.testing.assert_allclose(sig_theo, turb_df.std(axis=0),
                               atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, turb_df.mean(axis=0),
                               atol=0.01)


def test_con_iec_mn_std_dev():
    """mean and standard of iec turbulence, and that con'd turb is regen'd
    """
    # given -- constraining points
    con_spat_df = pd.DataFrame([['vxt', 'p0', 0.0, 0.0, 70.0]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': 70,
              'T': 300, 'dt': 0.5}
    coh_model, spc_model, wsp_model = 'iec', 'kaimal', 'iec'
    con_turb_df = gen_turb(con_spat_df, coh_model=coh_model,
                           spc_model=spc_model,
                           wsp_model=wsp_model, **kwargs)
    # given -- simulated, constrainted turbulence
    y, z = 0, 72
    sim_spat_df = gen_spat_grid(y, z)
    sig_theo = np.array([1.834, 1.834, 1.4672, 0.917])
    u_theo = np.array([-10, -10.05650077210035, 0, 0])

    # when
    sim_turb_df = gen_turb(sim_spat_df, con_data={'con_spat_df': con_spat_df,
                                                  'con_turb_df': con_turb_df},
                           coh_model=coh_model, spc_model=spc_model,
                           wsp_model=wsp_model, all_df=True, **kwargs)
    # then
    np.testing.assert_allclose(sig_theo, sim_turb_df.std(axis=0),
                               atol=0.01, rtol=0.50)  # std devs are close
    np.testing.assert_allclose(u_theo, sim_turb_df.mean(axis=0),
                               atol=0.01)  # means are close
    np.testing.assert_allclose(con_turb_df.vxt_p0,
                               sim_turb_df.vxt_p0,
                               atol=0.01)  # regen'd const pt matches const


def test_collocated_turb():
    """if simulation point is collocated with constraint
    """
    # given
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': 75,
              'T': 300, 'dt': .25}
    coh_model, spc_model, wsp_model = 'iec', 'kaimal', 'iec'
    con_spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = gen_turb(con_spat_df,
                           coh_model=coh_model, spc_model=spc_model,
                           wsp_model=wsp_model, **kwargs)
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 30],
                            ['vxt', 'p1', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    # when
    turb_df = gen_turb(spat_df, con_data={'con_spat_df': con_spat_df,
                                          'con_turb_df': con_turb_df},
                       coh_model=coh_model, spc_model=spc_model,
                       wsp_model=wsp_model, **kwargs)
    # then
    pd.testing.assert_series_equal(con_turb_df.vxt_p0, turb_df.vxt_p1,
                                   check_names=False)
