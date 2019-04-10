# -*- coding: utf-8 -*-
"""Test functions in wind_profiles.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb._utils import _spat_colnames
from pyconturb.core.spectral_models import kaimal_spectrum, get_spec_values


def test_get_spec_values_custom():
    """Custom spectrum is just k plus 1 for all f, y, and z"""
    # given
    spat_df = pd.DataFrame([[1, 0, 0, 0, 50],
                            [0, 0, 0, 0, 70]], columns=_spat_colnames)
    f = [0.5, 2.0]
    s_theory = np.array([[2, 1], [2, 1]])  # [s_u(0.5), s_v(0.5)], [s_u(2.0), s_v(2.0)]
    spec_func = lambda f, k, y, z: np.array([np.ones(np.size(f))*(k + 1)
                                             for k in spat_df.k]).T
    # when
    spc_np = get_spec_values(f, spat_df, spec_func)
    # then
    np.testing.assert_allclose(s_theory, spc_np, atol=1e-4)


def test_get_spec_values_kaimal():
    """"""
    # given
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 70]], columns=_spat_colnames)
    f, u_hub = [0.5, 2.0], 10
    s_theory = np.array([[0.0676126976, 0.1210076452],  # s_u(0.5), s_v(0.5))
                         [0.0068066176, 0.0124465662]])  # s_u(2.0), s_v(2.0))
    spc_func = kaimal_spectrum
    # when
    spc_np = get_spec_values(f, spat_df, spc_func, u_hub=u_hub)
    # then
    np.testing.assert_allclose(s_theory, spc_np, atol=1e-4)


def test_kaimal_spectrum_value():
    """Check the value for get_kaimal_spectrum"""
    # given
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 70]], columns=_spat_colnames)
    f = [0.5, 2.0]
    kwargs =  {'u_hub': 10}
    s_theory = np.array([[0.0676126976, 0.1210076452],  # s_u(0.5), s_v(0.5))
                         [0.0068066176, 0.0124465662]])  # s_u(2.0), s_v(2.0))
    # when (check input: series and np.array)
    spc_np1 = kaimal_spectrum(f, spat_df.k, spat_df.y, spat_df.z, **kwargs)
    spc_np2 = kaimal_spectrum(f, *(spat_df[['k', 'y', 'z']].values.T), **kwargs)
    # then
    np.testing.assert_allclose(s_theory, spc_np1, atol=1e-4)
    np.testing.assert_allclose(s_theory, spc_np2, atol=1e-4)


if __name__ == '__main__':
    test_get_spec_values_custom()
    test_get_spec_values_kaimal()
    test_kaimal_spectrum_value()
