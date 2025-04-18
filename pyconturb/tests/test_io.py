# -*- coding: utf-8 -*-
"""test io functions
"""
import os

import numpy as np
import pandas as pd

from pyconturb.io import df_to_h2turb, h2turb_to_df, df_to_bts, bts_to_df
from pyconturb._utils import gen_spat_grid


def test_pctdf_to_h2turb(tmp_path):
    """save PyConTurb dataframe as HAWC2 binary file and load again"""
    # given
    z = np.array([50, 70, 80])
    spat_df = gen_spat_grid(0, z)
    nt, dt = 100, 0.2
    t = np.arange(nt) * dt
    for dtype in [np.float32, np.float64]:
        turb_df = pd.DataFrame(np.random.rand(nt, 3*z.size),
                               index=t, dtype=dtype,
                               columns=[f'{c}_p{i}' for i in range(z.size) for c in 'uvw'])
        # when
        df_to_h2turb(turb_df, spat_df, tmp_path)
        test_df = h2turb_to_df(spat_df, tmp_path, nt=nt, dt=dt)
        # then
        pd.testing.assert_frame_equal(turb_df, test_df, check_dtype=False)


def test_pctdf_to_bts():
    """save PyConTurb dataframe as TurbSim binary file and load again"""
    # given
    paths = ['garbage', 'garbage.bts']
    spat_df = gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.random.rand(100, 6),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    uzhubs = [None, (0.5, 60)]
    # when
    for path in paths:  # check with and without extension
        for uzhub in uzhubs:
            df_to_bts(turb_df, spat_df, path, uzhub=uzhub)
            test_df = bts_to_df(path)
            os.remove('./garbage.bts')
            # then
            for c in 'uvw':  # pandas won't ignore column order, so use numpy instead
                turb_c_df = turb_df.filter(regex=f'{c}_')
                test_c_df = test_df.filter(regex=f'{c}_')
                np.testing.assert_allclose(turb_c_df, test_c_df, atol=1e-5,
                                           rtol=np.inf)  # just look at abs tol


def test_pctdf_to_bts_const(tmp_path):
    """check if signals are constant"""
    # given
    path = str(tmp_path / 'garbage.bts')
    spat_df = gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.ones((100, 6)),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    df_to_bts(turb_df, spat_df, path, uzhub=None)
    test_df = bts_to_df(path)
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
