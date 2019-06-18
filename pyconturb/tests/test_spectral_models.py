# -*- coding: utf-8 -*-
"""Test functions in wind_profiles.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb import TimeConstraint
from pyconturb.spectral_models import kaimal_spectrum, get_spec_values, data_spectrum
from pyconturb._utils import _spat_rownames


def test_get_spec_values_custom():
    """Custom spectrum is just k plus 1 for all f, y, and z"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [50, 70]], index=_spat_rownames)
    f = [0.5, 2.0]
    s_theory = np.array([[1, 2], [1, 2]])  # [s_u(0.5), s_v(0.5)], [s_u(2.0), s_v(2.0)]
    spec_func = lambda f, k, y, z: np.array([np.ones(np.size(f))*(k + 1)
                                             for k in spat_df.loc['k']]).T
    # when
    spc_np = get_spec_values(f, spat_df, spec_func)
    # then
    np.testing.assert_allclose(s_theory, spc_np, atol=1e-4)


def test_get_spec_values_kaimal():
    """"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [50, 70]], index=_spat_rownames)
    f, u_ref = [0.5, 2.0], 10
    s_theory = np.array([[0.0676126976, 0.1210076452],  # s_u(0.5), s_v(0.5))
                         [0.0068066176, 0.0124465662]])  # s_u(2.0), s_v(2.0))
    spc_func = kaimal_spectrum
    # when
    spc_np = get_spec_values(f, spat_df, spc_func, u_ref=u_ref)
    # then
    np.testing.assert_allclose(s_theory, spc_np, atol=1e-4)


def test_kaimal_spectrum_value():
    """Check the value for get_kaimal_spectrum"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [50, 70]], index=_spat_rownames)
    f = [0.5, 2.0]
    kwargs = {'u_ref': 10}
    s_theory = np.array([[0.0676126976, 0.1210076452],  # s_u(0.5), s_v(0.5))
                         [0.0068066176, 0.0124465662]])  # s_u(2.0), s_v(2.0))
    # when (check input: series and np.array)
    spc_np1 = kaimal_spectrum(f, spat_df.loc['k'], spat_df.loc['y'], spat_df.loc['z'], **kwargs)
    spc_np2 = kaimal_spectrum(f, *(spat_df.loc[['k', 'y', 'z']].values), **kwargs)
    # then
    np.testing.assert_allclose(s_theory, spc_np1, atol=1e-4)
    np.testing.assert_allclose(s_theory, spc_np2, atol=1e-4)


def test_data_spectrum():
    """verify the data interpolator implement in data_spectrum works + dtype"""
    # given
    f = [0, 0.5]
    k, y, z = np.repeat(range(3), 3), np.zeros(9, dtype=int), np.tile([40, 70, 100], 3)
    con_tc = TimeConstraint([[0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [50, 90, 50, 90, 50, 90],
                             [0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]],
                            index=['k', 'x', 'y', 'z', 0.0, 1.0],
                            columns=['u_p0', 'u_p1', 'v_p0', 'v_p1', 'w_p0', 'w_p1'])
    spc_theo = np.tile(np.interp([0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5],
                                 np.arange(6), [4, 16, 36, 64, 100, 144]),
                       (2, 1))
    # when
    spc_arr = data_spectrum(f, k, y, z, con_tc=con_tc)
    # then
    np.testing.assert_array_equal(spc_theo, spc_arr)


if __name__ == '__main__':
    test_get_spec_values_custom()
    test_get_spec_values_kaimal()
    test_kaimal_spectrum_value()
    test_data_spectrum()
