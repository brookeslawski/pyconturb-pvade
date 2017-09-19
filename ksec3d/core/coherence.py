# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""

import numpy as np


def get_coherence(k1, k2, x1, x2, freq,
                  coh_model='rinker', **kwargs):
    """KSEC-3D coherence function.
    """
    return


def get_iec_coherence(k1, k2, x1, x2, freq,
                      ed=3, **kwargs):
    """Exponential coherence specified in IEC 61400-1

    Parameters
    ----------
    k1 : int
        Index of turbulence component at location 1. 0=u, 1=v, 2=w.
    k2 : int
        Index of turbulence component at location 2.
    x1 : array_like
        3D spatial location of point 1.
    x2 : array_like
        3D spatial location of point 2.
    freq : array_like
        Frequencies for which to evaluate coherence.
    kwargs : dictionary
        Other variables specific to this coherence model.
    """

    x1 = np.asarray(x1)  # convert everything to arrays in case they weren't
    x2 = np.asarray(x2)  # passed in as arrays
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)

    if ed != 3:
        raise ValueError('Only edition 3 is permitted.')

    return
