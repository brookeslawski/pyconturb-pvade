# -*- coding: utf-8 -*-
"""Test functions for power spectra
"""
import numpy as np
import pandas as pd
import pytest

from pyconturb import TimeConstraint
from pyconturb.spectral_models import kaimal_spectrum, get_spec_values, data_spectrum
from pyconturb._utils import _spat_rownames

su_0p5 = 0.0676126976  # theoretical Su(0.5 Hz) for Kaimal, u_ref = 10, z > 60
sv_0p5 = 0.1210076452  # Sv(0.5) for Kaimal, u_ref = 10, z > 60
su_2p0 = 0.0068066176  # Su(2.0) for Kaimal, u_ref = 10, z > 60
sv_2p0 = 0.0124465662  # Sv(2.0) for Kaimal, u_ref = 10, z > 60


def test_get_spec_values_custom():
    """Custom spectrum is just k plus 1 for all f, y, and z"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [50, 70]], index=_spat_rownames)
    f = [0.5, 2.0]
    s_theory = np.array([[1, 2], [1, 2]])  # [s_u(0.5), s_v(0.5)], [s_u(2.0), s_v(2.0)]
    spec_func = lambda f, spat_df: np.array([np.ones(np.size(f))*(k + 1)
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
    s_theory = np.array([[su_0p5, sv_0p5],  # s_u(0.5), s_v(0.5))
                         [su_2p0, sv_2p0]])  # s_u(2.0), s_v(2.0))
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
    s_theory = np.array([[su_0p5, sv_0p5],  # s_u(0.5), s_v(0.5))
                         [su_2p0, sv_2p0]])  # s_u(2.0), s_v(2.0))
    # when (check input: series and np.array)
    spc_np1 = kaimal_spectrum(f, spat_df, **kwargs)
    spc_np2 = kaimal_spectrum(f, spat_df, **kwargs)
    # then
    np.testing.assert_allclose(s_theory, spc_np1, atol=1e-4)
    np.testing.assert_allclose(s_theory, spc_np2, atol=1e-4)


def test_data_spectrum():
    """verify 1) data interpolator implement in data_spectrum works, 2) dtype"""
    # given
    f, T, dt = [0, 0.5], 2, 1
    # k, y, z = , , 
    spat_df = pd.DataFrame([np.repeat(range(3), 3), np.zeros(9), np.zeros(9, dtype=int),
                            np.tile([40, 70, 100], 3)], index=_spat_rownames)
    con_tc = TimeConstraint([[0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [50, 90, 50, 90, 50, 90],
                             [0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]],
                            index=['k', 'x', 'y', 'z', 0.0, 1.0])
    spec_cols = 2 * np.abs(np.fft.rfft(con_tc.get_time(), axis=0)/(T/dt))**2 * T
    spec_theo = np.tile(np.interp([0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5],
                                  np.arange(6), spec_cols[0]), (2, 1))
    # when
    spec_arr = data_spectrum(f, spat_df, con_tc=con_tc)
    # then
    np.testing.assert_array_equal(spec_theo, spec_arr)


def test_data_spectrum_nolabels():
    """test 1)can interpolate with no/incorrect column labels 2) float comparisons"""
    # given
    dt, T = 1, 4
    spat_df = pd.DataFrame([[0], [0], [0], [65]], index=_spat_rownames)
    con_tc = TimeConstraint([[0, 0], [0, 0], [0, 0], [60, 70],
                             [9, 17], [11, 20], [10, 18], [9, 16]],
                            index=['k', 'x', 'y', 'z', 0, dt, 2*dt, 3*dt],
                            columns=['dog', 'cat'])
    mags_theo = np.abs(np.fft.rfft(con_tc.get_time(), axis=0)/(T/dt))
    spec_theo = np.mean(2*mags_theo**2*T, axis=1).reshape(-1, 1)
    f = np.arange(mags_theo.shape[0]) / T + 1e-12
    # when
    spec = data_spectrum(f, spat_df, con_tc=con_tc)
    # then
    np.testing.assert_array_almost_equal(spec, spec_theo)


def test_data_spectrum_missingcon():
    """Test (1) spectra with no constraint are zero (2) a warning is raised"""
    # given
    f, u_ref = 0.5, 10
    spat_df = pd.DataFrame([[0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [60, 70, 60, 70, 60, 70]], index=['k', 'x', 'y', 'z'])
    con_tc = TimeConstraint([[0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0],
                              [50, 90, 50, 90],
                              [0, 0, 0, 0], [1, 2, 3, 4]],
                            index=['k', 'x', 'y', 'z', 0.0, 1.0])
    spec_theo = np.array([[1.75, 2.5, sv_0p5, sv_0p5, 10.75, 12.5]])  # 1x6 array of theory
    # when
    with pytest.warns(Warning):
        spec_arr = data_spectrum(f, spat_df, con_tc=con_tc, u_ref=u_ref)
    # then
    np.testing.assert_array_almost_equal(spec_theo, spec_arr)


if __name__ == '__main__':
    test_get_spec_values_custom()
    test_get_spec_values_kaimal()
    test_kaimal_spectrum_value()
    test_data_spectrum()
    test_data_spectrum_nolabels()
    test_data_spectrum_missingcon()
