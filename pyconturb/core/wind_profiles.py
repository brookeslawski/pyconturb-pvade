# -*- coding: utf-8 -*-
"""Functions related to definitions of mean wind speed profiles
"""
import numpy as np


def get_wsp_values(spat_df, wsp_func, ignore_k=False, **kwargs):
    """Get (n_sim,) array of mean wind speed values.
    U = wsp_func(y, z, **kwargs)."""
    wsp_values = wsp_func(spat_df.y, spat_df.z, **kwargs)
    if not ignore_k:  # all components except u go to zero
        wsp_values *= (spat_df.k == 0)
    return np.array(wsp_values)  # convert to array in case series given


def constant_profile(**kwargs_prof):
    """constant mean wind speed, default is zero"""
    kwargs_prof = {**{'u_const': 0}, **kwargs_prof}

    def wsp_func(y, z, **kwargs):
        return np.ones_like(y) * kwargs_prof['u_const']

    return wsp_func


def power_profile(**kwargs_prof):
    """Returns a function that takes height and returns power-law mean wind speed"""

    def wsp_func(y, z, **kwargs):
        return kwargs_prof['u_hub'] * (z / kwargs_prof['z_hub']) ** kwargs_prof['alpha']

    return wsp_func
