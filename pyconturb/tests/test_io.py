# -*- coding: utf-8 -*-
"""test io functions
"""
import os

import numpy as np
import pandas as pd

from pyconturb.io import df_to_h2turb, h2turb_to_df
from pyconturb._utils import gen_spat_grid


def test_pctdf_to_h2turb():
    """save PyConTurb dataframe as binary file and load again"""
    # given
    path = '.'
    spat_df = gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.random.rand(100, 6),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    df_to_h2turb(turb_df, spat_df, '.')
    test_df = h2turb_to_df(spat_df, path)
    [os.remove(os.path.join('.', f'{c}.bin')) for c in 'uvw']
    # then
    pd.testing.assert_frame_equal(turb_df, test_df, check_dtype=False)



if __name__ == '__main__':
    test_pctdf_to_h2turb()
