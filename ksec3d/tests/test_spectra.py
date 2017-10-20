# -*- coding: utf-8 -*-
"""Test functions in spectra.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from ksec3d.core.spectra import get_kaimal_spectrum


def test_kaimal_value():
    """Check the value for get_kaimal_spectrum
    """
    # given
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50],
                            ['vyt', 'p0', 0, 0, 70]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    freq = [0.5, 2.0]
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3}
    # s_u(0.5, 2.0) = [0.2274190946, 0.0228944394]
    # s_v(0.5, 2.0) = [0.26042826, 0.0267869083]
    s_theory = np.array([[0.2274190946, 0.26042826],  # 0.5 Hz
                         [0.0228944394, 0.0267869083]])  # 2.0 Hz

    # when
    spc_df = get_kaimal_spectrum(spat_df, freq,
                                 **kwargs)

    # then
    np.testing.assert_allclose(s_theory, spc_df, atol=1e-4)
