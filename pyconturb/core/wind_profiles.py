# -*- coding: utf-8 -*-
"""Functions related to definitions of mean wind speed profiles
"""
import warnings

import numpy as np


_iec_alpha = 0.2  # power law coefficient from IEC 61400-1 Ed. 3


def get_wsp_profile(spat_df, con_data=None, wsp_model='iec', **kwargs):
    """get the mean wind speed profile at each point

    Notes
    -----
    The mean wind speeds are defined according to the x, y, z coordinate system
    in the HAWC2 turbulence coordinate system. In particular, x is directed
    upwind, z is vertical up, and y is lateral to form a right-handed
    coordinate system. Thus, any non-zero mean wind speeds are often negative,
    since positive x is directed upwind.
    """

    if wsp_model == 'none':  # no mean wind speed
        wsp_prof = np.zeros(spat_df.shape[0])

    elif wsp_model == 'iec':  # IEC Ed. 3 power law
        if 'ed' not in kwargs.keys():
            warnings.warn('No IEC edition specified -- assuming Ed. 3')
            kwargs['ed'] = 3
        if kwargs['ed'] != 3:  # only allow edition 3
            raise ValueError('Only edition 3 is permitted.')
        if any([k not in kwargs.keys() for k in ['v_hub', 'z_hub']]):
            raise ValueError('Missing keyword arguments for IEC mean ' +
                             'wind profile model')
        wsp_prof = (kwargs['v_hub'] *
                    (spat_df.mask(spat_df.k != 0, other=0).z
                    / kwargs['z_hub']) ** _iec_alpha).values

    elif wsp_model == 'data':  # interpolate from data
        # check that we're given data to calculate profile from
        if con_data is None:
            raise ValueError('No data given for profile interpolation!')

        # pull out data heights and mean wind speed profile
        con_spat_df = con_data['con_spat_df']
        con_turb_df = con_data['con_turb_df']
        zs = con_spat_df[con_spat_df.k == 0].z.values
        mwsps = con_turb_df.filter(regex='u_', axis=1).mean(axis=0).values
        mwsps = mwsps[np.argsort(zs)].astype(float)  # sort for later interp
        zs = zs[np.argsort(zs)].astype(float)  # sort for later interp

        # get max and min measurement heights and corresponding wsp
        z_min, z_max = zs.min(), zs.max()  # lowest and highest msnt hts
        u_min, u_max = mwsps[zs.argmin()], mwsps[zs.argmax()]  # mean wsps

        # create profile: interp btwn msmnt heights, power law otherwise
        z_sims = spat_df.mask(spat_df.k != 0, other=0).z.values
        wsp_prof = np.zeros(z_sims.size)  # initialize profile to zeros
        wsp_prof[z_sims <= z_min] = u_min * (z_sims[z_sims <= z_min] /
                                             z_min) ** _iec_alpha  # power law
        wsp_prof[z_sims >= z_max] = u_max * (z_sims[z_sims >= z_max] /
                                             z_max) ** _iec_alpha  # power law
        wsp_prof[(z_min < z_sims) & (z_sims < z_max)] = \
            np.interp(z_sims[(z_min < z_sims) & (z_sims < z_max)],
                      zs, mwsps)  # interp between msmnt heights

    else:  # unknown profile model
        raise ValueError(f'Wind profile model "{wsp_model}" not recognized.')

    return wsp_prof
