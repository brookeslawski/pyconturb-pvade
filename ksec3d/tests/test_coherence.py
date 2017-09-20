# -*- coding: utf-8 -*-
"""Test functions in coherence.py
"""

import numpy as np
from py.test import raises

from ksec3d.core.coherence import get_coherence, get_iec_coherence


def test_main_default():
    """Check the default value for get_coherence
    """
    # given
    k1, k2 = 0, 0
    x1, x2 = [0, 0, 0], [0, 1, 0]
    freq = 1
    kwargs = {'v_hub': 1, 'l_c': 1}
    coh_theory = 5.637379774e-6

    # when
    coh = get_coherence(k1, k2, x1, x2, freq,
                        **kwargs)

    # then
    assert np.isclose(coh, coh_theory)


def test_main_badcohmodel():
    """Should raise an error if a wrong coherence model is passed in
    """
    # given
    k1, k2 = 0, 0
    x1, x2 = [0, 0, 0], [1, 1, 1]
    freq = 1

    # when & then
    with raises(ValueError):
        get_coherence(k1, k2, x1, x2, freq,
                      coh_model='garbage')


def test_iec_badedition():
    """IEC coherence should raise an error if any edn other than 3 is given
    """
    # given
    k1, k2 = 0, 0
    x1, x2 = [0, 0, 0], [1, 1, 1]
    freq = 1
    kwargs = {'ed': 4, 'v_hub': 12, 'l_c': 340.2}

    # when & then
    with raises(ValueError):
        get_iec_coherence(k1, k2, x1, x2, freq,
                          **kwargs)


def test_iec_missingkwargs():
    """IEC coherence should raise an error if missing parameter(s)
    """
    # given
    k1, k2 = 0, 0
    x1, x2 = [0, 0, 0], [1, 1, 1]
    freq = 1
    kwargs = {'ed': 3, 'v_hub': 12}

    # when & then
    with raises(ValueError):
        get_iec_coherence(k1, k2, x1, x2, freq,
                          **kwargs)


def test_iec_value():
    """Verify that the value of IEC coherence matches theory
    """
    # given
    k1, k2 = 0, 0
    x1, x2 = [0, 0, 0], [0, 1, 0]
    freq = 0.5
    kwargs = {'ed': 3, 'v_hub': 2, 'l_c': 3}
    coh_theory = 0.0479231144

    # when
    coh = get_iec_coherence(k1, k2, x1, x2, freq,
                            **kwargs)

    # then
    assert np.isclose(coh, coh_theory)

    # given
    k1, k2 = 2, 2
    x1, x2 = [0, 0, 0], [0, 1, 0]
    freq = 1
    kwargs = {'ed': 3, 'v_hub': 2, 'l_c': 3}
    coh_theory = 0

    # when
    coh = get_iec_coherence(k1, k2, x1, x2, freq,
                            **kwargs)

    # then
    assert np.isclose(coh, coh_theory)
