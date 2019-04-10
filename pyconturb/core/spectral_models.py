# -*- coding: utf-8 -*-
"""Functions related to definitions of mean wind speed profiles
"""
import numpy as np


def get_spec_values(f, spat_df, spec_func, **kwargs):
    """Get the (nf, n_sp) array of spectra values"""
    return spec_func(f, spat_df.k, spat_df.y, spat_df.z, **kwargs)


def kaimal_spectrum(f, k, y, z, **kwargs):
    """f is (nf,); k, y and z are (n_sp,), u_hub is float or int. returns (nf, n_sp,).
    No std scaling -- that's done with the magnitudes."""
    k, y, z = [np.asarray(x) for x in (k, y, z)]  # in case pd.series passed in
    f = np.reshape(f, (-1, 1))  # convert to column array
    lambda_1 = 0.7 * z * (z < 60) + 42 * (z >= 60)  # length scale changes with z
    l_k = lambda_1 * (8.1 * (k == 0) + 2.7 * (k == 1) + 0.66 * (k == 2))
    tau = np.reshape((l_k / kwargs['u_hub']), (1, -1))  # L_k / U. row vector
    spc_arr = (4 * tau) / np.power(1. + 6 * tau * f, 5. / 3.)  # Kaimal 1972
    return spc_arr.astype(float)  # pandas causes object issues, ensure float
