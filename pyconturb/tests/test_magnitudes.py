# -*- coding: utf-8 -*-
"""Test functions in magnitudes.py

Author
------
Jenni Rinker
rink@dtu.dk
"""
import numpy as np
import pandas as pd

from pyconturb.magnitudes import spc_to_mag, get_magnitudes
from pyconturb.sig_models import iec_sig
from pyconturb.spectral_models import kaimal_spectrum
from pyconturb._utils import _spat_rownames, get_freq


def test_get_mags_iec():
    """verify iec magnitudes"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [90, 90]], index=_spat_rownames)
    kwargs = {'T': 2, 'dt': 1, 'turb_class': 'a', 'z_ref': 90, 'u_ref': 10, 'alpha': 0.2}
    sig_func, spec_func = iec_sig, kaimal_spectrum
    mags_theo = [[0, 0], [2.096, 1.6768]]
    t, f = get_freq(**kwargs)
    # when
    mags_arr = get_magnitudes(spat_df, spec_func, sig_func, **kwargs)
    # then
    np.testing.assert_almost_equal(mags_arr, mags_theo)


def test_spc_to_mag():
    """verify we get mags with correct std out from function"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [90, 90]], index=_spat_rownames)
    spc_arr = np.ones((10, 2))
    kwargs = {'T': 10, 'dt': 1}
    t, f = get_freq(**kwargs)
    sig_func = lambda k, y, z, **kwargs: 0.1 + 0.1 * np.array(k)
    std_theo = [0.1, 0.2]
    # when
    mags = spc_to_mag(spat_df, spc_arr, sig_func, **kwargs)
    std_sim = np.std(np.fft.irfft(mags, n=t.size, axis=0) * t.size, axis=0)
    # then
    np.testing.assert_almost_equal(std_sim, std_theo)


if __name__ == '__main__':
    test_get_mags_iec()
    test_spc_to_mag()
