# -*- coding: utf-8 -*-
"""Test functions in helpers.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb.core.helpers import gen_spat_grid, get_iec_sigk, \
                                    spat_to_pair_df, combine_spat_df,\
                                    h2t_to_uvw


def test_gen_spat_grid():
    """Test the generation of the spatial grid
    """
    # given
    y = [-10, 10]
    z = [50, 70]
    first_row = ['vxt', 'p0', 0, -10, 50]
    theo_size = (3 * len(y) * len(z)) * len(first_row)

    # when
    spat_df = gen_spat_grid(y, z)

    # then
    assert spat_df.size == theo_size
    assert all(spat_df.iloc[0, :2] == first_row[:2])
    assert np.array_equal(spat_df.iloc[0, 2:].values,
                          np.array(first_row[2:]))


def test_get_iec_sigk():
    """verify values for iec sig_k
    """
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50],
                            ['vyt', 'p0', 0, 0, 50],
                            ['vzt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2}
    sig_k = get_iec_sigk(spat_df, **kwargs)
    sig_theo = [1.834, 1.4672, 0.917]
    np.testing.assert_allclose(sig_k, sig_theo)


def test_spat_to_pair_df():
    """converting spat_df to pair_df
    """
    # given
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50],
                            ['vyt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    pair_df_theo = pd.DataFrame([['vxt', 0, 0, 50,
                                  'vyt', 0, 0, 50]],
                                columns=['k1', 'x1', 'y1', 'z1',
                                         'k2', 'x2', 'y2', 'z2'])
    # when
    pair_df = spat_to_pair_df(spat_df)
    # then
    pd.testing.assert_frame_equal(pair_df, pair_df_theo, check_dtype=False)


def test_one_spat_to_pair():
    """spatial df with one element should produce an empty pair
    """
    # given
    spat_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    pair_df_theo = pd.DataFrame(np.empty((0, 8)),
                                columns=['k1', 'x1', 'y1', 'z1',
                                         'k2', 'x2', 'y2', 'z2'])
    # when
    pair_df = spat_to_pair_df(spat_df)
    # then
    pd.testing.assert_frame_equal(pair_df, pair_df_theo, check_dtype=False)


def test_combine_spat_df():
    """combining two spat_df
    """
    # given
    left_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    right_df = pd.DataFrame([['vxt', 'p0', 0, 0, 60]],
                            columns=['k', 'p_id', 'x', 'y', 'z'])
    comb_df_theo = pd.DataFrame([['vxt', 'p0', 0, 0, 50],
                                 ['vxt', 'p1', 0, 0, 60]],
                                columns=['k', 'p_id', 'x', 'y', 'z'])
    # when
    comb_df = combine_spat_df(left_df, right_df)
    # then
    pd.testing.assert_frame_equal(comb_df, comb_df_theo, check_dtype=False)


def test_combine_empty_spat_dfs():
    """combining empty spat_dfs
    """
    # given
    left_df = pd.DataFrame([['vxt', 'p0', 0, 0, 50]],
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    right_df = pd.DataFrame([['vxt', 'p0', 0, 0, 60]],
                            columns=['k', 'p_id', 'x', 'y', 'z'])
    empty_df = pd.DataFrame(np.empty((0, 5)),
                            columns=['k', 'p_id', 'x', 'y', 'z'])
    theo_dfs = [left_df, right_df]
    # when
    comb_dfs = [combine_spat_df(left_df, empty_df),
                combine_spat_df(empty_df, right_df)]
    # then
    for (res_df, theo_df) in zip(comb_dfs, theo_dfs):
        pd.testing.assert_frame_equal(res_df, theo_df, check_dtype=False)


def test_h2t_to_uvw():
    """test converting turb_df to uvw coor sys
    """
    # given
    turb_df = pd.DataFrame([[1, 1, 1]], index=[1],
                           columns=['vxt_p0', 'vyt_p0', 'vzt_p0'])
    theo_df = pd.DataFrame([[-1, -1, 1]], index=[1],
                           columns=['u_p0', 'v_p0', 'w_p0'])
    # when
    uvw_turb_df = h2t_to_uvw(turb_df)
    # then
    pd.testing.assert_frame_equal(uvw_turb_df, theo_df,
                                  check_dtype=False)
