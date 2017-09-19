# -*- coding: utf-8 -*-
"""Functions related to definitions of spectral models
"""

import numpy as np


def get_kaimal_spectrum(f, tau, sig):
    """ Kaimal spectrum (continuous, one-sided) for frequency f and time
        length scale tau = L/U.
        Args:
            f (numpy array): frequencies
            tau (float/numpy array): integral time scale (L/U)
            sig (float): standard deviation

        Returns:
            S (numpy array): Kaimal spectrum evaluated at f, tau, sig
    """

    S = (sig**2) * (4. * tau) / \
        np.power(1. + 6.*f*tau, 5. / 3.)            # Kaimal 1972

    return S
