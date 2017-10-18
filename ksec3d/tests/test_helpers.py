# -*- coding: utf-8 -*-
"""Test functions in helpers.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from ksec3d.core.helpers import gen_spat_grid, get_iec_sigk


def test_gen_spat_grid():
    """Test the generation of the spatial grid
    """
    # given
    x = [-10, 10]
    z = [50, 70]
    first_row = ['vxt', 'p0', 0, -10, 50]
    theo_size = (3 * len(x) * len(z)) * len(first_row)

    # when
    spat_df = gen_spat_grid(x, z)

    # then
    assert spat_df.size == theo_size
    assert all(spat_df.iloc[0, :2] == first_row[:2])
    assert np.array_equal(spat_df.iloc[0, 2:].values,
                          np.array(first_row[2:]))


def test_get_iec_sigk():
    """verify values for iec sig_k
    """
    spat_df = pd.DataFrame([['vxt', 0, 0, 50],
                            ['vyt', 0, 0, 50],
                            ['vzt', 0, 0, 50]],
                           columns=['k', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2}
    sig_k = get_iec_sigk(spat_df, **kwargs)
    sig_theo = [1.834, 1.4672, 0.917]
    np.testing.assert_allclose(sig_k, sig_theo)
