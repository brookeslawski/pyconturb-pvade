# -*- coding: utf-8 -*-
"""Test functions in coherence.py
"""

import numpy as np

from ksec3d.core.helpers import gen_spat_grid


def test_gen_spat_grid():
    """Test the generation of the spatial grid
    """
    # given
    x = [-10, 10]
    z = [50, 70]
    first_row = ['u', 'p0', 0, -10, 50]
    theo_size = (3 * len(x) * len(z)) * len(first_row)

    # when
    spat_df = gen_spat_grid(x, z)

    # then
    assert spat_df.size == theo_size
    assert all(spat_df.iloc[0, :2] == first_row[:2])
    assert np.array_equal(spat_df.iloc[0, 2:].values,
                          np.array(first_row[2:]))
