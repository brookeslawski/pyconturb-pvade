# -*- coding: utf-8 -*-
"""Functions related to definitions of mean wind speed profiles
"""
import numpy as np

from pyconturb.core.wind_profiles import power_profile


def get_sig_values(spat_df, sig_func, **kwargs):
    """Get (n_sim,) array of std dev values.
    ti_arr = ti_func(k, y, z, **kwargs)."""
    ti_values = sig_func(spat_df.k, spat_df.y, spat_df.z, **kwargs)
    return ti_values


def iec_sig(k, y, z, **kwargs):
    """get numpy array of iec std dev values"""
    assert kwargs['turb_class'].lower() in 'abc', 'Invalid or no turbulence class!'
    i_ref = {'a': 0.16, 'b': 0.14, 'c': 0.12}[kwargs['turb_class'].lower()]
    sig1 = i_ref * (0.75 * kwargs['u_hub'] + 5.6)  # std dev
    sig_k = sig1 * np.asarray(1.0 * (k == 0) + 0.8 * (k == 1) + 0.5 * (k == 2))
    return np.array(sig_k, dtype=float)
