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
                          np.repeat(xs.reshape(-1), ks.size),
                          np.repeat(ys.reshape(-1), ks.size),
                          np.repeat(zs.reshape(-1), ks.size))).T
    spat_df = pd.DataFrame(spat_arr,
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    spat_df[['x', 'y', 'z']] = spat_df[['x', 'y', 'z']].astype(float)
    return spat_df
