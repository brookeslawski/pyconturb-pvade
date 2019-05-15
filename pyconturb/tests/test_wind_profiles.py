# -*- coding: utf-8 -*-
"""test functions
"""

import numpy as np
import pandas as pd

from pyconturb.wind_profiles import constant_profile, power_profile, get_wsp_values
from pyconturb._utils import _spat_colnames


def test_get_mean_wsp_custom():
    """Verify correct profile for custom profile (constant)"""
    # given
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 50],
                            [2, 0, 0, 0, 50],
                            [0, 1, 0, 0, 90]], columns=_spat_colnames)
    wsp_func = lambda y, z, **kwargs: 4  # constant wind speed
    u_theory = np.array([4, 0, 0, 4])
    # when
    wsp_vals = get_wsp_values(spat_df, wsp_func)
    # then
    np.testing.assert_allclose(u_theory, wsp_vals)


def test_get_mean_wsp_pwr():
    """Verify correct profile for power law"""
    # given
    kwargs = {'u_ref': 10, 'z_ref': 90, 'alpha': 0.2}
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 50],
                            [2, 0, 0, 0, 50],
                            [0, 1, 0, 0, 90]], columns=_spat_colnames)
    u_theory = np.array([8.890895361, 0, 0, 10])
    wsp_func = power_profile
    # when
    wsp_vals = get_wsp_values(spat_df, wsp_func, **kwargs)
    # then
    np.testing.assert_allclose(u_theory, wsp_vals)


def test_power_profile():
    """Verify power law profile"""
    # given
    kwargs = {'u_ref': 10, 'z_ref': 90, 'alpha': 0.2}
    y, z = np.array([0, 0]), np.array([50, 90])
    u_theory = [8.890895361, 10]
    # when
    wsp_func = power_profile(y, z, **kwargs)
    # then
    np.testing.assert_allclose(u_theory, wsp_func)


def test_constant_profile():
    """Verify power law profile"""
    # given
    y, z = np.array([0, 0]), np.array([50, 90])
    u_theory = [[0, 0], [4, 4]]
    kwargs = {'u_ref': 4}
    # when
    wsp_prof_0 = constant_profile(y, z)
    wsp_prof_c = constant_profile(y, z, **kwargs)
    # then
    np.testing.assert_allclose(u_theory[0], wsp_prof_0)
    np.testing.assert_allclose(u_theory[1], wsp_prof_c)


if __name__ == '__main__':
    test_get_mean_wsp_custom()
    test_get_mean_wsp_pwr()
    test_power_profile()
    test_constant_profile()
