# -*- coding: utf-8 -*-
"""test io functions
"""
import os

import numpy as np
import pandas as pd

from pyconturb.io import df_to_h2turb, h2turb_to_df, df_to_bts, bts_to_df
from pyconturb._utils import gen_spat_grid


def test_pctdf_to_h2turb():
    """save PyConTurb dataframe as HAWC2 binary file and load again"""
    # given
    path = '.'
    spat_df = gen_spat_grid(0, [50, 70])
    nt, dt = 100, 0.2
    t = np.arange(nt) * dt
    turb_df = pd.DataFrame(np.random.rand(nt, 6),
                           index=t,
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    df_to_h2turb(turb_df, spat_df, '.')
    test_df = h2turb_to_df(spat_df, path, nt=nt, dt=dt)
    [os.remove(os.path.join('.', f'{c}.bin')) for c in 'uvw']
    # then
    pd.testing.assert_frame_equal(turb_df, test_df, check_dtype=False)


def test_pctdf_to_bts():
    """save PyConTurb dataframe as TurbSim binary file and load again"""
    # given
    paths = ['garbage', 'garbage.bts']
    spat_df = gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.random.rand(100, 6),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    yzhubs = [None, (0, 50)]
    # when
    for path in paths:  # check with and without extension
        for yzhub in yzhubs:
            df_to_bts(turb_df, spat_df, path, yzhub=yzhub)
            test_df = bts_to_df(path)
            os.remove('./garbage.bts')
            # then
            for c in 'uvw':  # pandas won't ignore column order, so use numpy instead
                turb_c_df = turb_df.filter(regex=f'{c}_')
                test_c_df = test_df.filter(regex=f'{c}_')
                np.testing.assert_allclose(turb_c_df, test_c_df, atol=1e-5,
                                           rtol=np.inf)  # just look at abs tol


def test_pctdf_to_bts_const():
    """check if signals are constant"""
    # given
    path = 'garbage.bts'
    spat_df = gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.ones((100, 6)),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    df_to_bts(turb_df, spat_df, path, yzhub=None)
    test_df = bts_to_df(path)
    os.remove('./garbage.bts')
    # then
    for c in 'uvw':  # pandas won't ignore column order, so use numpy instead
        turb_c_df = turb_df.filter(regex=f'{c}_')
        test_c_df = test_df.filter(regex=f'{c}_')
        np.testing.assert_allclose(turb_c_df, test_c_df, atol=1e-5,
                                   rtol=np.inf)  # just look at abs tol


if __name__ == '__main__':
    test_pctdf_to_h2turb()
    test_pctdf_to_bts()
    test_pctdf_to_bts_const()
