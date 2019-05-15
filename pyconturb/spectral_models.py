# -*- coding: utf-8 -*-
"""Define how the turbulence power spectrum varies with k, y and z.

You can either use the built-in models (see below) or create your own function.
Note that these spectra are continuous, one-sided spectra. The conversion from
continuous to discrete spectra and scaling to the appropriate variance is done
during simulation.
"""
import numpy as np

from pyconturb._utils import _DEF_KWARGS


def get_spec_values(f, spat_df, spec_func, **kwargs):
    """Power spectral density (PSD) for points/components in ``spat_df``.

    The ``spec_func`` must be a function of the form::

        spec_values = spec_func(f, k, y, z, **kwargs)

    where f, k, y and z can be floats, np.arrays or pandas.Series. You can use
    the functions built into PyConTurb (see below) or define your own custom
    function. The output is assumed to be in (m^2/s^2)/Hz = m^2/s. There is no
    need to scale to the correct variance -- the spectrum is scaled during simulation
    in order to produce the standard deviation specified by ``sig_func``.

    Parameters
    ----------
    f : array-like
        [Hz] Frequency(s) for which PSD is to be calculated. Size is ``n_f``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, p_id, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    spec_func : function
        Function to map k, y and z to the continuous, one-sided power spectral
        density in m^2/s.
    **kwargs
        Keyword arguments to pass into ``spec_func``.

    Returns
    -------
    spec_values : np.array
        [m/s] PSD values for the given spatial locations(s)/component(s).
        Dimension is ``(n_f, n_sp,)``.
    """
    return spec_func(f, spat_df.k, spat_df.y, spat_df.z, **kwargs)


def kaimal_spectrum(f, k, y, z, u_ref=_DEF_KWARGS['u_ref'], **kwargs):
    """Kaimal PSD as specified in IEC 61400-1 Ed. 3.
    f is (nf,); k, y and z are (n_sp,), u_ref is float or int. returns (nf, n_sp,).
    No std scaling -- that's done with the magnitudes.

    Parameters
    ----------
    f : array-like
        [Hz] Frequency(s) for which PSD is to be calculated. Size is ``n_f``.
    k : array-like
        [-] Integer indicator of the turbulence component. 0=u, 1=v, 2=w.
    y : array-like
        [m] Location of point(s) in the lateral direction. Can be int/float,
        np.array or pandas.Series.
    z : array-like
        [m] Location of point(s) in the vertical direction. Can be int/float,
        np.array or pandas.Series.
    u_ref : int/float, optional
        [m/s] Mean wind speed at reference height.
    **kwargs
        Unused (optional) keyword arguments. 

    Returns
    -------
    sig_values : np.array
        [m/s] Turbulence standard deviation(s) at the specified location(s).
    """
    kwargs = {**{'u_ref': u_ref}, **kwargs}  # add dflts if not given
    k, y, z = [np.asarray(x) for x in (k, y, z)]  # in case pd.series passed in
    f = np.reshape(f, (-1, 1))  # convert to column array
    lambda_1 = 0.7 * z * (z < 60) + 42 * (z >= 60)  # length scale changes with z
    l_k = lambda_1 * (8.1 * (k == 0) + 2.7 * (k == 1) + 0.66 * (k == 2))
    tau = np.reshape((l_k / kwargs['u_ref']), (1, -1))  # L_k / U. row vector
    spc_arr = (4 * tau) / np.power(1. + 6 * tau * f, 5. / 3.)  # Kaimal 1972
    return spc_arr.astype(float)  # pandas causes object issues, ensure float
