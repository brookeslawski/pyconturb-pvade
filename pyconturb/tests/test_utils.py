# -*- coding: utf-8 -*-
"""test util functions
"""
import os

import numpy as np
import pandas as pd

import pyconturb._utils as utils


_spat_colnames = utils._spat_colnames


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
        comb_df = utils.combine_spat_df(df1, df2)
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
    spat_df = utils.gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.random.rand(100, 6),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    utils.df_to_h2turb(turb_df, spat_df, '.')
    test_df = utils.h2turb_to_df(spat_df, path)
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
    spat_df = utils.gen_spat_grid(y, z)
    # then
    pd.testing.assert_frame_equal(theo_df, spat_df, check_dtype=False)


def test_get_freq_values():
    """verify correct output of get_freq"""
    # given
    kwargs = {'T': 41, 'dt': 2}
    t_theo = np.arange(0, 42, 2)
    f_theo = np.arange(0, 11) / 41
    # when
    t_out, f_out = utils.get_freq(**kwargs)
    # then
    np.testing.assert_almost_equal(t_theo, t_out)
    np.testing.assert_almost_equal(f_theo, f_out)


def test_make_hawc2_input():
    """verify correct strings for hawc2 input"""
    # given
    turb_dir = '.'
    spat_df = utils.gen_spat_grid([-10, 10], [109, 129])
    kwargs = {'z_ref': 119, 'T': 600, 'dt': 1, 'u_ref': 10}
    str_cntr_theo = '  center_pos0             0.0 0.0 -119.0 ; hub height\n'
    str_mann_theo = ('  begin mann ;\n    filename_u ./u.bin ; \n'
                     + '    filename_v ./v.bin ; \n    filename_w ./w.bin ; \n'
                     + '    box_dim_u 600 10.0 ; \n    box_dim_v 2 20.0 ; \n'
                     + '    box_dim_w 2 20.0 ; \n    dont_scale 1 ; \n  end mann ')
    str_output_theo = '  wind free_wind 1 10.0 0.0 -109.0 # wind_p0 ; '
    # when
    str_cntr, str_mann, str_output = \
        utils.make_hawc2_input(turb_dir, spat_df, **kwargs)
    # then
    assert(str_cntr == str_cntr_theo)
    assert(str_mann == str_mann_theo)
    assert(str_output.split('\n')[0] == str_output_theo)


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
        uvw = utils.rotate_time_series(*xyz)
        # then
        np.testing.assert_almost_equal(uvw[0], uvw_theo[0])


if __name__ == '__main__':
    test_combine_spat_df()
    test_pctdf_to_h2turb()
    test_gen_spat_grid()
    test_get_freq_values()
#    test_h2t_to_uvw()
    test_make_hawc2_input()
    test_rotate_time_series()
