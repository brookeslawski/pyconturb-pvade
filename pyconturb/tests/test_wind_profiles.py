# -*- coding: utf-8 -*-
"""Test functions in wind_profiles.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb.core.wind_profiles import get_wsp_profile


def test_get_wsp_profile_iec():
    """Check the iec value for get_wsp_profile
    """
    # given
    spat_df = pd.DataFrame([['vxt', 0, 0, 50],
                            ['vyt', 0, 0, 50],
                            ['vzt', 0, 0, 50],
                            ['vxt', 0, 0, 90]],
                           columns=['k', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'z_hub': 90}
    u_theory = np.array([-8.890895361, 0, 0, -10])

    # when
    wsp_profile = get_wsp_profile(spat_df, wsp_model='iec', **kwargs)

    # then
    np.testing.assert_allclose(u_theory, wsp_profile)


def test_get_wsp_profile_data():
    """Check the data value for get_wsp_profile
    """
    # given
    con_spat_df = pd.DataFrame([['vxt', 0, 0, 10],
                                ['vxt', 0, 0, 20],
                                ['vyt', 0, 0, 20]],
                               columns=['k', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[-10, -20, 0]], index=[0],
                               columns=['vxt_p0', 'vxt_p1', 'vyt_p1'])
    spat_df = pd.DataFrame([['vxt', 0, 0, 5],
                            ['vxt', 0, 0, 15],
                            ['vxt', 0, 0, 25]],
                           columns=['k', 'x', 'y', 'z'])
    u_theory = np.array([-8.705505633, -15, -20.91279105])

    # when
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    wsp_profile = get_wsp_profile(spat_df, con_data=con_data,
                                  wsp_model='data')

    # then
    np.testing.assert_allclose(u_theory, wsp_profile)
