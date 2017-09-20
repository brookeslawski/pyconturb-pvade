# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""

import numpy as np


def get_coherence(k1, k2, x1, x2, freq,
                  coh_model='iec', **kwargs):
    """KSEC-3D coherence function

    Calls coherence-model-specific subfunctions.

    Notes
    -----
    The spatial coordinate system is x positive to the right, z positive up,
    and y positive downwind.

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
    coh_model : str, optional
        Coherence model to use. Default is IEC Ed. 3.
    """

    if coh_model == 'iec':  # IEC coherence model
        if 'ed' not in kwargs.keys():  # add IEC ed to kwargs if not passed in
            kwargs['ed'] = 3
        coh = get_iec_coherence(k1, k2, x1, x2, freq, **kwargs)

    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized.')

    return coh


def get_iec_coherence(k1, k2, x1, x2, freq,
                      **kwargs):
    """Exponential coherence specified in IEC 61400-1

    Notes
    -----
    The spatial coordinate system is x positive to the right, z positive up,
    and y positive downwind.

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

    x1 = np.asarray(x1)  # convert locations to arrays in case they weren't
    x2 = np.asarray(x2)  # passed in as such

    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['v_hub', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')

    r = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)  # inter-pt distance
    if (k1 == 0) and (k2 == 0):  # u at point 1 and u at point 2
        coh = np.exp(-12 * np.sqrt((freq * r / kwargs['v_hub'])**2 +
                                   (0.12 * r / kwargs['l_c'])**2))
    else:  # any other combination of turbulence components is no coherence
        coh = 0

    return coh
