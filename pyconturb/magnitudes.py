# -*- coding: utf-8 -*-
"""Functions related to definitions of spectral models
"""
import numpy as np

from pyconturb.spectral_models import get_spec_values
from pyconturb.sig_models import get_sig_values
from pyconturb._utils import get_freq


def get_magnitudes(spat_df, spec_func, sig_func, **kwargs):
    """"""
    t, freq = get_freq(**kwargs)
    spc_arr = get_spec_values(freq, spat_df, spec_func, **kwargs)
    mags_arr = spc_to_mag(spat_df, spc_arr, sig_func, **kwargs)
    return mags_arr


def spc_to_mag(spat_df, spc_arr, sig_func, **kwargs):
    """Convert spectral array to magnitudes (sets DC component to 0)
    """
    # get unscaled magnitudes
    t, freq = get_freq(**kwargs)
    n_t, df = t.size, freq[1]
    mags_arr = np.sqrt(spc_arr * df / 2)  # (nf, nsp)
    mags_arr[0, :] = 0.  # mean is zero
    # scale to get the correct ti
    std_arr = get_sig_values(spat_df, sig_func, **kwargs)  # (n_sp,)
    alpha = std_arr / np.std(np.fft.irfft(mags_arr, n=n_t, axis=0) * n_t, axis=0)
    return (alpha * mags_arr).astype(float)  # (nf, nsp)
