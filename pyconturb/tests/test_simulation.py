# -*- coding: utf-8 -*-
"""Test functions in simulation.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb.simulation import gen_turb
from pyconturb.wind_profiles import constant_profile
from pyconturb._utils import gen_spat_grid, _spat_colnames


def test_iec_turb_mn_std_dev():
    """test that iec turbulence has correct mean and std deviation"""
    # given
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_hub': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_hub': 70, 'T': 300, 'dt': 1}
    sig_theo = np.array([1.834, 1.4672, 0.917, 1.834, 1.4672, 0.917])
    u_theo = np.array([10, 0, 0, 10.27066087, 0, 0])
    # when
    turb_df = gen_turb(spat_df, **kwargs)
    # then
    np.testing.assert_allclose(sig_theo, turb_df.std(axis=0), atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, turb_df.mean(axis=0),  atol=0.01)


def test_con_iec_mn_std_dev():
    """mean and standard of iec turbulence, and that con'd turb is regen'd
    """
    # given -- constraining points
    con_spat_df = pd.DataFrame([[0, 0, 0, 0, 70]], columns=_spat_colnames)
    kwargs = {'u_hub': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_hub': 70, 'T': 300, 'dt': 0.5,}
    coh_model = 'iec'
    con_turb_df = gen_turb(con_spat_df, coh_model=coh_model, **kwargs)
    # given -- simulated, constrainted turbulence
    y, z = 0, [70, 72]
    sim_spat_df = gen_spat_grid(y, z)
    sig_theo = np.tile([1.834, 1.4672, 0.917], 2)  # sig_u, sig_v, sig_w
    u_theo = np.array([10, 0, 0, 10.05650077210035, 0, 0])  # U1, ... U2, ...
    # when
    sim_turb_df = gen_turb(sim_spat_df, con_data={'con_spat_df': con_spat_df,
                                                  'con_turb_df': con_turb_df},
                           coh_model=coh_model, **kwargs)
    # then (std dev, mean, and regen'd time series should be close)
    np.testing.assert_allclose(sig_theo, sim_turb_df.std(axis=0), atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, sim_turb_df.mean(axis=0), atol=0.01)
    np.testing.assert_allclose(con_turb_df.u_p0, sim_turb_df.u_p0, atol=0.01)


def test_collocated_turb():
    """if simulation point is collocated with constraint"""
    # given
    kwargs = {'u_hub': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_hub': 75, 'T': 300, 'dt': .25}
    coh_model = 'iec'
    con_spat_df = pd.DataFrame([[0, 0, 0, 0, 50]], columns=_spat_colnames)
    con_turb_df = gen_turb(con_spat_df, coh_model=coh_model,  **kwargs)
    spat_df = pd.DataFrame([[0, 0, 0, 0, 30],
                            [0, 1, 0, 0, 50]], columns=_spat_colnames)
    theory = con_turb_df.u_p0 - con_turb_df.u_p0.mean()
    wsp_func = constant_profile  # zero profile
    # when
    turb_df = gen_turb(spat_df, con_data={'con_spat_df': con_spat_df,
                                          'con_turb_df': con_turb_df},
                       wsp_func=wsp_func, coh_model=coh_model, **kwargs)
    test = turb_df.u_p1
    # then
    pd.testing.assert_series_equal(theory, test, check_names=False)


if __name__ == '__main__':
    test_iec_turb_mn_std_dev()
    test_con_iec_mn_std_dev()
    test_collocated_turb()
