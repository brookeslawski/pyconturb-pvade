# -*- coding: utf-8 -*-
"""Test functions for mean wind speed profile
"""

import numpy as np
import pandas as pd
import pytest

from pyconturb import TimeConstraint, gen_spat_grid
from pyconturb.wind_profiles import constant_profile, power_profile, get_wsp_values, \
    data_profile
from pyconturb._utils import _spat_rownames


def test_get_mean_wsp_custom():
    """Verify correct profile for custom profile (constant)"""
    # given
    spat_df = pd.DataFrame([[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [50, 50, 50, 90]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0', 'u_p1'])
    wsp_func = lambda spat_df, **kwargs: 4 * (spat_df.loc['k'] == 0)  # constant wind speed, only u
    u_theory = np.array([4, 0, 0, 4])
    # when
    wsp_vals = get_wsp_values(spat_df, wsp_func)
    # then
    np.testing.assert_allclose(u_theory, wsp_vals)


def test_get_mean_wsp_pwr():
    """Verify correct profile for power law"""
    # given
    kwargs = {'u_ref': 10, 'z_ref': 90, 'alpha': 0.2}
    spat_df = pd.DataFrame([[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [50, 50, 50, 90]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0', 'u_p1'])
    u_theory = np.array([8.890895361, 0, 0, 10])
    wsp_func = power_profile
    # when
    wsp_vals = get_wsp_values(spat_df, wsp_func, **kwargs)
    # then
    np.testing.assert_allclose(u_theory, wsp_vals)


def test_power_profile():
    """Verify power law profile for all components"""
    # given
    kwargs = {'u_ref': 10, 'z_ref': 90, 'alpha': 0.2}
    spat_df = gen_spat_grid(0, [50, 90], comps=[0, 1, 2])
    u_theory = [8.890895361, 0, 0, 10, 0, 0]
    # when
    wsp_func = power_profile(spat_df, **kwargs)
    # then
    np.testing.assert_allclose(u_theory, wsp_func)


def test_constant_profile():
    """Verify constant profile for all components"""
    # given
    spat_df = gen_spat_grid(0, [50, 90], comps=[0, 1, 2])
    u_theory = [[0, 0, 0, 0, 0, 0], [4, 0, 0, 4, 0, 0]]
    kwargs = {'u_ref': 4}
    # when
    wsp_prof_0 = constant_profile(spat_df)
    wsp_prof_c = constant_profile(spat_df, **kwargs)
    # then
    np.testing.assert_allclose(u_theory[0], wsp_prof_0)
    np.testing.assert_allclose(u_theory[1], wsp_prof_c)


def test_data_profile():
    """verify profile interpolated from data for all components"""
    # given
    spat_df = gen_spat_grid(0, [40, 70, 100], comps=[0, 1, 2])
    con_tc = TimeConstraint([[0, 0, 1, 1, 2, 2],  # k
                             [0, 0, 0, 0, 0, 0],  # x
                             [0, 0, 0, 0, 0, 0],  # y
                             [50, 90, 50, 90, 50, 90],  # z
                             [8, 10, -1, 1, 0, 1]],
                            index=['k', 'x', 'y', 'z', 0.0],
                            columns=['u_p0', 'u_p1', 'v_p0', 'v_p1', 'w_p0', 'w_p1'])
    u_theo = [8, -1, 0, 9, 0, 0.5, 10, 1, 1]
    # when
    wsp_prof = data_profile(spat_df, con_tc)
    # then
    np.testing.assert_allclose(u_theo, wsp_prof)


def test_data_profile_nocon():
    """verify (1) warning is raised with no con in u (2) iec is returned for u and 0 for w"""
    # given
    u_ref, z_ref = 10, 70
    spat_df = gen_spat_grid(0, [40, 70, 100], comps=[0, 2])
    con_tc = TimeConstraint([[1, 1], [0, 0], [0, 0], [50, 90], [8, 10]],
                            index=['k', 'x', 'y', 'z', 0.0],
                            columns=['v_p0', 'v_p1'])
    u_theo = np.zeros(6)
    u_theo[::2] = 10*(np.array([40, 70, 100])/70)**0.2  # default to iec theory
    # when
    with pytest.warns(Warning):
        wsp_prof = data_profile(spat_df, con_tc, u_ref=u_ref, z_ref=z_ref)
    # then
    np.testing.assert_allclose(u_theo, wsp_prof)

if __name__ == '__main__':
    test_get_mean_wsp_custom()
    test_get_mean_wsp_pwr()
    test_power_profile()
    test_constant_profile()
    test_data_profile()
    test_data_profile_nocon()
