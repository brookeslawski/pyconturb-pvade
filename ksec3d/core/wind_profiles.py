# -*- coding: utf-8 -*-
"""Functions related to definitions of mean wind speed profiles
"""
import warnings

import numpy as np


def get_wsp_profile(spat_df,
                    wsp_model='iec', **kwargs):
    """get the mean wind speed profile at each point

    Notes
    -----
    The mean wind speeds are defined according to the x, y, z coordinate system
    in the HAWC2 turbulence coordinate system. In particular, x is directed
    upwind, z is vertical up, and y is lateral to form a right-handed
    coordinate system. Thus, any non-negative mean wind speeds are often
    negative, since positive x is directed upwind.
    """

    if wsp_model == 'none':  # no mean wind speed
        wsp_prof = np.zeros(spat_df.shape[0])

    elif wsp_model == 'iec':  # Kaimal spectral model
        alpha = 0.2  # iec ed 3 power law coefficient
        if 'ed' not in kwargs.keys():
            warnings.warn('No IEC edition specified -- assuming Ed. 3')
            kwargs['ed'] = 3
        if kwargs['ed'] != 3:  # only allow edition 3
            raise ValueError('Only edition 3 is permitted.')
        if any([k not in kwargs.keys() for k in ['v_hub', 'z_hub']]):
            raise ValueError('Missing keyword arguments for IEC mean ' +
                             'wind profile model')
        wsp_prof = -(kwargs['v_hub'] *
                     (spat_df.mask(spat_df.k != 'u', other=0).z
                     / kwargs['z_hub']) ** alpha).values

    else:  # unknown profile model
        raise ValueError(f'Wind profile model "{wsp_model}" not recognized.')

    return wsp_prof
