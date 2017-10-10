# -*- coding: utf-8 -*-
"""Functions related to definitions of mean wind speed profiles
"""
import warnings


def get_wsp_profile(spat_df,
                    wsp_model='iec', **kwargs):
    """get the mean wind speed profile at each point
    """

    if wsp_model == 'iec':  # Kaimal spectral model
        alpha = 0.2  # power law coefficient
        if 'ed' not in kwargs.keys():
            warnings.warn('No IEC edition specified -- assuming Ed. 3')
            kwargs['ed'] = 3
        if kwargs['ed'] != 3:  # only allow edition 3
            raise ValueError('Only edition 3 is permitted.')
        if any([k not in kwargs.keys() for k in ['v_hub', 'z_hub']]):
            raise ValueError('Missing keyword arguments for IEC mean ' +
                             'wind profile model')
        wsp_df = kwargs['v_hub'] * (spat_df.z / kwargs['z_hub']) ** alpha

    else:  # unknown coherence model
        raise ValueError(f'Wind profile model "{wsp_model}" not recognized.')

    return wsp_df.values
