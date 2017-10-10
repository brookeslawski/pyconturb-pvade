# -*- coding: utf-8 -*-
"""Test functions in coherence.py
"""

import numpy as np
import pandas as pd

from ksec3d.core.spectra import get_kaimal_spectrum


def test_kaimal_value():
    """Check the value for get_kaimal_spectrum
    """
    # given
    spat_df = pd.DataFrame([['u', 0, 0, 50],
                            ['u', 0, 0, 70]],
                           columns=['k', 'x', 'y', 'z'])
    freq = [0.5, 2.0]
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3}
    s_theory = 0.2020424997

    # when
    spc_df = get_kaimal_spectrum(spat_df, freq,
                                 **kwargs)

    # then
    assert np.isclose(spc_df.iloc[1, 0], s_theory)
