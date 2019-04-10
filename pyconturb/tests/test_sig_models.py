# -*- coding: utf-8 -*-
"""test functions
"""
import numpy as np
import pandas as pd
import pytest

from pyconturb.core.sig_models import get_sig_values, iec_sig
from pyconturb._utils import _spat_colnames


def test_get_sig_custom():
    """verify correct sig for a custom sig model"""
    # given
    kwargs = {'val': 3}
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 50],
                            [2, 0, 0, 0, 50],
                            [0, 1, 0, 0, 90]], columns=_spat_colnames)
    sig_func = lambda k, y, z, **kwargs: z * kwargs['val']
    sig_theory = [150, 150, 150, 270]
    # when
    sig_arr = get_sig_values(spat_df, sig_func, **kwargs)
    # then
    np.testing.assert_allclose(sig_theory, sig_arr)


def test_get_sig_iec():
    """verify correct sig for iec"""
    # given
    kwargs = {'turb_class': 'A', 'u_hub': 10, 'z_hub': 90, 'alpha': 0.2}
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 50],
                            [2, 0, 0, 0, 50],
                            [0, 1, 0, 0, 90]], columns=_spat_colnames)
    sig_func = iec_sig
    sig_theory = [2.096, 1.6768, 1.048, 2.096]
    # when
    sig_arr = get_sig_values(spat_df, sig_func, **kwargs)
    # then
    np.testing.assert_allclose(sig_theory, sig_arr)


def test_iec_sig_bad_tclass():
    """iec_sig should raise an error when a bad turbulence class is given"""
    with pytest.raises(AssertionError):
        iec_sig(0, 1, 2, turb_class='R')
        iec_sig(0, 1, 2, turb_class='cab')


def test_iec_sig_value():
    """verify the correct numbers are being produced"""
    # given
    kwargs = {'turb_class': 'A', 'u_hub': 10, 'z_hub': 90, 'alpha': 0.2}
    k, y, z = np.array([2, 1, 0]), np.array([0, 0, 0]), np.array([50, 50, 90])
    sig_theory = [1.048, 1.6768, 2.096]
    # when
    sig_arr = iec_sig(k, y, z, **kwargs)
    # then
    np.testing.assert_allclose(sig_theory, sig_arr)


if __name__ == '__main__':
    test_get_sig_custom()
    test_get_sig_iec()
    test_iec_sig_bad_tclass()
    test_iec_sig_value()
