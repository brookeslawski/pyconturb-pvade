# -*- coding: utf-8 -*-
"""Miscellaneous helper functions

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd


def gen_spat_grid(y, z):
    """Generate spat_df (all turbulent components and grid defined by x and z)
    """
    ys, zs = np.meshgrid(y, z)
    ks = np.array(['u', 'v', 'w'])
    xs = np.zeros_like(ys)
    ps = [f'p{i:.0f}' for i in np.arange(xs.size)]
    spat_arr = np.vstack((np.tile(ks, xs.size),
                          np.repeat(ps, ks.size),
                          np.repeat(xs.T.reshape(-1), ks.size),
                          np.repeat(ys.T.reshape(-1), ks.size),
                          np.repeat(zs.T.reshape(-1), ks.size))).T
    spat_df = pd.DataFrame(spat_arr,
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    spat_df[['x', 'y', 'z']] = spat_df[['x', 'y', 'z']].astype(float)
    return spat_df


def get_iec_sigk(spat_df, **kwargs):
    """get sig_k for iec
    """
    sig = kwargs['i_ref'] * (0.75 * kwargs['v_hub'] + 5.6)  # std dev
    sig_k = sig * (1.0 * (spat_df.k == 'u') + 0.8 * (spat_df.k == 'v') +
                   0.5 * (spat_df.k == 'w')).values
    return sig_k
