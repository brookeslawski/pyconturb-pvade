# -*- coding: utf-8 -*-
"""Test functions in magnitudes.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

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


def test_get_data_mags_zinterp():
    """Check vertical interpolation for data magnitudes
    """
    # given
    d = 1.0
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 60]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    con_spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50],
                                ['vxt', 'p1', 0, 0, 70]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[1, 2],
                                [2, 4],
                                [1.5, 3]], index=d * np.arange(3),
                               columns=['vxt_p0', 'vxt_p1'])
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    freq = np.fft.rfftfreq(con_turb_df.shape[0], d=d)
    kwargs = {'method': 'z_interp'}
    mags_theo = pd.DataFrame([[6.750+0.j], [-1.125-0.64951905j]],
                             index=freq, columns=['vxt_p0'])
    # when
    mags_df = get_data_magnitudes(spat_df, freq, con_data, **kwargs)
    # then
    np.testing.assert_allclose(mags_theo, mags_df)
