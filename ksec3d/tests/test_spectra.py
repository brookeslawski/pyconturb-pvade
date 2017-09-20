# -*- coding: utf-8 -*-
"""Test functions in coherence.py
"""

import numpy as np

from ksec3d.core.spectra import get_kaimal_spectrum


def test_kaimal_value():
    """Check the value for get_kaimal_spectrum
    """
    # given
    f, tau, sig = 1, 2, 3
    s_theory = 1.001752056

    # when
    s = get_kaimal_spectrum(f, tau, sig)

    # then
    assert np.isclose(s, s_theory)
