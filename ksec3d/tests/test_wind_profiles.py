# -*- coding: utf-8 -*-
"""Test functions in coherence.py
"""

import numpy as np
import pandas as pd

from ksec3d.core.wind_profiles import get_wsp_profile


def test_get_wsp_profile():
    """Check the value for get_wsp_profile
    """
    # given
    spat_df = pd.DataFrame([['u', 0, 0, 50],
                            ['v', 0, 0, 50],
                            ['w', 0, 0, 50],
                            ['u', 0, 0, 90]],
                           columns=['k', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'z_hub': 90}
    u_theory = np.array([8.890895361, 0, 0, 10])

    # when
    wsp_profile = get_wsp_profile(spat_df, wsp_model='iec', **kwargs)

    # then
    np.testing.assert_allclose(u_theory, wsp_profile)
