# -*- coding: utf-8 -*-
"""Test functions in magnitudes.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd
import pytest

from pyconturb.core.magnitudes import get_kaimal_spectrum, get_data_magnitudes


def test_kaimal_value():
    """Check the value for get_kaimal_spectrum
    """
    # given
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50],
                            ['vyt', 'p0', 0, 0, 70]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    freq = [0.5, 2.0]
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3}
    # s_u(0.5, 2.0) = [0.2274190946, 0.0228944394]  # reminder to self
    # s_v(0.5, 2.0) = [0.26042826, 0.0267869083]  # reminder to self
    s_theory = np.array([[0.2274190946, 0.26042826],  # 0.5 Hz
                         [0.0228944394, 0.0267869083]])  # 2.0 Hz

    # when
    spc_df = get_kaimal_spectrum(spat_df, freq,
                                 **kwargs)

    # then
    np.testing.assert_allclose(s_theory, spc_df, atol=1e-4)


def test_get_data_mags_err_bad_data():
    """raise a ValueError when there is a component to interpolate with no data
    """
    # given
    d = 1.0
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    con_spat_df = pd.DataFrame([['vyt', 'p0', 0, 0, 50]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[1], [2], [1.5]], index=d * np.arange(3),
                               columns=['vyt_p0'])
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    freq = np.fft.rfftfreq(con_turb_df.shape[0], d=d)
    kwargs = {'method': 'z_interp'}
    # when & then
    with pytest.raises(ValueError):
        get_data_magnitudes(spat_df, freq, con_data, **kwargs)


def test_get_data_mags_err_bad_method():
    """raise a ValueError when invalid method specified
    """
    # given
    d = 1.0
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    con_spat_df = pd.DataFrame([['vyt', 'p0', 0, 0, 50]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[1], [2], [1.5]], index=d * np.arange(3),
                               columns=['vyt_p0'])
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    freq = np.fft.rfftfreq(con_turb_df.shape[0], d=d)
    kwargs = {'method': 'garbage'}
    # when & then
    with pytest.raises(ValueError):
        get_data_magnitudes(spat_df, freq, con_data, **kwargs)


def test_get_data_mags_zinterp():
    """Check vertical interpolation for data magnitudes
    """
    # given
    d = 1.0
    spat_df = pd.DataFrame([['vxt', 'p0', 0, -10, 80],
                            ['vxt', 'p1', 0, 0, 60],
                            ['vxt', 'p2', 0, 0, 70],
                            ['vxt', 'p3', 0, 5, 50],
                            ['vxt', 'p4', 0, 10, 40]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    con_spat_df = pd.DataFrame([['vxt', 'p0', 0, -10, 50],
                                ['vxt', 'p1', 0, -5, 70],
                                ['vxt', 'p2', 0, 5, 50],
                                ['vxt', 'p3', 0, 5, 70]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[0, 2, 2, 4]], index=[d],
                               columns=['vxt_p0', 'vxt_p1',
                                        'vxt_p2', 'vxt_p3'])
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    freq = np.fft.rfftfreq(con_turb_df.shape[0], d=d)
    kwargs = {'method': 'z_interp'}
    mags_theo = pd.DataFrame([[3, 2, 3, 1, 1]],
                             index=freq, columns=['vxt_p0', 'vxt_p1',
                                                  'vxt_p2', 'vxt_p3',
                                                  'vxt_p4'],
                             dtype=np.float)
    # when
    mags_df = get_data_magnitudes(spat_df, freq, con_data, **kwargs)
    # then
    pd.testing.assert_frame_equal(mags_theo, mags_df)
