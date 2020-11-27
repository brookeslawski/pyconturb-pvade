# -*- coding: utf-8 -*-
"""Test functions for standard deviation
"""
import numpy as np
import pandas as pd
import pytest

from pyconturb.core import TimeConstraint
from pyconturb.sig_models import get_sig_values, iec_sig, data_sig
from pyconturb._utils import _spat_rownames


def test_get_sig_custom():
    """verify correct sig for a custom sig model"""
    # given
    kwargs = {'val': 3}
    spat_df = pd.DataFrame([[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [50, 50, 50, 90]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0', 'u_p1'])
    sig_func = lambda spat_df, **kwargs: spat_df.loc['z'] * kwargs['val']
    sig_theory = [150, 150, 150, 270]
    # when
    sig_arr = get_sig_values(spat_df, sig_func, **kwargs)
    # then
    np.testing.assert_allclose(sig_theory, sig_arr)


def test_get_sig_iec():
    """verify correct sig for iec"""
    # given
    kwargs = {'turb_class': 'A', 'u_ref': 10, 'z_ref': 90, 'alpha': 0.2}
    spat_df = pd.DataFrame([[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [50, 50, 50, 90]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0', 'u_p1'])
    sig_func = iec_sig
    sig_theory = [2.096, 1.6768, 1.048, 2.096]
    # when
    sig_arr = get_sig_values(spat_df, sig_func, **kwargs)
    # then
    np.testing.assert_allclose(sig_theory, sig_arr)


def test_iec_sig_bad_tclass():
    """iec_sig should raise an error when a bad turbulence class is given"""
    with pytest.raises(AssertionError):
        iec_sig(0, turb_class='R')
        iec_sig(0, turb_class='cab')


def test_iec_sig_value():
    """verify the correct numbers are being produced"""
    # given
    kwargs = {'turb_class': 'A', 'u_ref': 10, 'z_ref': 90, 'alpha': 0.2}
    spat_df = pd.DataFrame([[2, 1, 0], [0, 0, 0], [0, 0, 0], [50, 50, 90]], index=_spat_rownames)
    sig_theory = [1.048, 1.6768, 2.096]
    # when
    sig_arr = iec_sig(spat_df, **kwargs)
    # then
    np.testing.assert_allclose(sig_theory, sig_arr)


def test_data_sig():
    """verify 1) data interpolator in data_sig works, 2) dtype 3) bad columns"""
    # given
    k, y, z = np.repeat(range(3), 3), np.zeros(9, dtype=int), np.tile([40, 70, 100], 3)
    spat_df = pd.DataFrame([k, np.zeros(9), y, z], index=_spat_rownames)
    con_tc = TimeConstraint([[0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                             [50, 90, 50, 90, 50, 90],
                             [0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]],
                            index=['k', 'x', 'y', 'z', 0.0, 1.0],
                            columns=['a', 'b', 'c', 'd', 'e', 'f'])
    sig_theo = [0.5, 0.75, 1., 1.5, 1.75, 2., 2.5, 2.75, 3.]
    # when
    sig_arr = data_sig(spat_df, con_tc)
    # then
    np.testing.assert_allclose(sig_theo, sig_arr)


def test_data_sig_nocon():
    """verify 1) a warning is raised and 2) fxn returns iec for comps with no con"""
    # given
    u_ref = 10
    k, y, z = np.repeat(range(3), 3), np.zeros(9, dtype=int), np.tile([40, 70, 100], 3)
    spat_df = pd.DataFrame([k, np.zeros(9), y, z], index=_spat_rownames)
    con_tc = TimeConstraint([[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                              [50, 90, 50, 90], [0, 0, 0, 0], [1, 2, 3, 4]],
                            index=['k', 'x', 'y', 'z', 0.0, 1.0])
    sig_theo = [0.5, 0.75, 1., 1.5, 1.75, 2., 1.048, 1.048, 1.048]
    # when
    with pytest.warns(Warning):
        sig_arr = data_sig(spat_df, con_tc, u_ref=u_ref)
    # then
    np.testing.assert_allclose(sig_theo, sig_arr)


if __name__ == '__main__':
    test_get_sig_custom()
    test_get_sig_iec()
    test_iec_sig_bad_tclass()
    test_iec_sig_value()
    test_data_sig()
    test_data_sig_nocon()
