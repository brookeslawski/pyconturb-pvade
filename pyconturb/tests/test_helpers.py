# -*- coding: utf-8 -*-
"""Test functions in helpers.py

Author
------
Jenni Rinker
rink@dtu.dk
"""
import os

import numpy as np
import pandas as pd

import pyconturb.core.helpers as pcth


_spat_colnames = pcth._spat_colnames


def test_combine_spat_df():
    """combine two spat_dfs. checks adding three types of dataframes: empty, 1-row with
    incorrect p_id and 2 rows with correct p_id.
    """
    # given
    dfa = pd.DataFrame([[0, 3, 0, 0, 60]], columns=_spat_colnames)
    dfb = pd.DataFrame([[0, 0, 0, 0, 50],
                        [0, 1, 0, 0, 60]], columns=_spat_colnames)
    empt_df = pd.DataFrame([], columns=_spat_colnames)
    frames = [empt_df, dfa, dfb]
    dfa_fixed = pd.DataFrame([[0, 0, 0, 0, 60]], columns=_spat_colnames)
    theo_frames = [empt_df, dfa_fixed, dfb]
    for i, (df1, df2) in enumerate([(a, b) for a in frames for b in frames]):  # df pairs
        # when
        comb_df = pcth.combine_spat_df(df1, df2)
        if i < 3:  # empty frame -> theo frames
            theo_df = theo_frames[i]
        elif i == 5:  # dfa on top, rows get flipped
            theo_df = pd.DataFrame([[0, 0, 0, 0, 60],
                                    [0, 1, 0, 0, 50]], columns=_spat_colnames)
        elif i > 5:  # dfb on bottom -> dfb
            theo_df = theo_frames[2]
        else:  # otherwise [dfa+dfa or dfa+empty] -> fixed dfa
            theo_df = theo_frames[1]
        # then
        pd.testing.assert_frame_equal(comb_df, theo_df, check_dtype=False,
                                      check_index_type=False)


def test_pctdf_to_h2turb():
    """save PyConTurb dataframe as binary file and load again"""
    # given
    path = '.'
    spat_df = pcth.gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.random.rand(100, 6),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    pcth.df_to_h2turb(turb_df, spat_df, '.')
    test_df = pcth.h2turb_to_df(spat_df, path)
    [os.remove(os.path.join('.', f'{c}.bin')) for c in 'uvw']
    # then
    pd.testing.assert_frame_equal(turb_df, test_df, check_dtype=False)


def test_gen_spat_grid():
    """verify column names and entries of spat grid
    """
    # given
    y, z = 0, 0
    theo_df = pd.DataFrame(np.zeros((3, 5)),  columns=_spat_colnames)
    theo_df.iloc[:, 0] = range(3)
    # when
    spat_df = pcth.gen_spat_grid(y, z)
    # then
    pd.testing.assert_frame_equal(theo_df, spat_df, check_dtype=False)


def test_get_iec_sigk():
    """verify values for iec sig_k
    """
    # given
    spat_df = pd.DataFrame([[0, 0, 0, 0, 50],
                            [1, 0, 0, 0, 50],
                            [2, 0, 0, 0, 50]], columns=_spat_colnames)
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2}
    sig_theo = [1.834, 1.4672, 0.917]
    # when
    sig_k = pcth.get_iec_sigk(spat_df, **kwargs)
    # then
    np.testing.assert_allclose(sig_k, sig_theo)


def test_h2t_to_uvw():
    """test converting turb_df to uvw coor sys
    """
    # given
    turb_df = pd.DataFrame([[1, 1, 1]], index=[1], columns=[f'v{c}t_p0' for c in 'xyz'])
    theo_df = pd.DataFrame([[-1, -1, 1]], index=[1], columns=['u_p0', 'v_p0', 'w_p0'])
    # when
    uvw_turb_df = pcth.h2t_to_uvw(turb_df)
    # then
    pd.testing.assert_frame_equal(uvw_turb_df, theo_df, check_dtype=False)


def test_rotate_time_series():
    """verify time series rotation"""
    # given
    xyzs = [[np.array([np.nan]), ] * 3,  # all nans
            [np.ones([1]), np.ones([1]), np.ones([1])],  # equal in all three
            [-np.ones([1]), np.ones([1]), np.ones([1])]]  # negative in x
    uvws = [xyzs[0],
            [np.sqrt(3) * np.ones([1]), np.zeros([1]), np.zeros([1])],
            [np.sqrt(3) * np.ones([1]), np.zeros([1]), np.zeros([1])]]
    # when
    for xyz, uvw_theo in zip(xyzs, uvws):
        uvw = pcth.rotate_time_series(*xyz)
        # then
        np.testing.assert_almost_equal(uvw[0], uvw_theo[0])
