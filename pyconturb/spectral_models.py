# -*- coding: utf-8 -*-
"""Define how the turbulence power spectrum varies with k, y and z.

You can either use the built-in models (see below) or create your own function.
Note that these spectra are continuous, one-sided spectra. The conversion from
continuous to discrete spectra and scaling to the appropriate variance is done
during simulation.
"""
import warnings

import numpy as np

from pyconturb._utils import _DEF_KWARGS, interpolator


def get_spec_values(f, spat_df, spec_func, **kwargs):
    """Power spectral density (PSD) for points/components in ``spat_df``.

    The ``spec_func`` must be a function of the form::

        spec_values = spec_func(f, spat_df, **kwargs).

    You can use the functions built into PyConTurb (see below) or define your own custom
    function. The output is assumed to be in (m^2/s^2)/Hz = m^2/s. There is no
    need to scale to the correct variance -- the spectrum is scaled during simulation
    in order to produce the standard deviation specified by ``sig_func``.

    Parameters
    ----------
    f : array-like
        [Hz] Frequency(s) for which PSD is to be calculated. Size is ``n_f``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, x, y, z]``, and each of the ``n_sp`` rows corresponds
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
        [m^2/s] PSD values for the given spatial locations(s)/component(s).
        Dimension is ``(n_f, n_sp,)``.
    """
    return spec_func(f, spat_df, **kwargs)


def data_spectrum(f, spat_df, con_tc=None, warn_datacon=True, **kwargs):
    """Power spectrum interpolated from a TimeConstraint object.

    See the Examples and/or Reference Guide for details on the interpolator logic or for
    how to construct a TimeConstraint object. Note that this function can only return
    frequency values as defined by T, dt in ``con_tc``.

    Note! If a component is requested for which there is no constraint, then this
    function will try to use the IEC spectra instead. Use the `warn_datacon` option to 
    disable the warning about this.

    Parameters
    ----------
    f : array-like
        [Hz] Frequency(s) for which PSD is to be calculated. Size is ``n_f``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    con_tc : pyconturb.TimeConstraint
        [-] Constraint object. Must have correct format; see documentation on
        PyConTurb's TimeConstraint object for more details.
    warn_datacon : boolean
        [-] Warn if a requested component does not have a constraint, which results in
        an attempt at using the Kaimal spectrum. Default is True.
    **kwargs
        Unused (optional) keyword arguments.

    Returns
    -------
    spec_values : np.array
        [m^2/s] PSD values for the given spatial locations(s)/component(s).
        Dimension is ``(n_f, n_sp,)``.
    """
    if con_tc is None:
        raise ValueError('No data provided!')
    # get time array and isolate/sanitize useful variables
    time_df = con_tc.get_time()
    df, nt = 1 / (time_df.index[-1] + time_df.index[1]), time_df.shape[0]
    f_reqs = np.array(f, ndmin=1)
    k, y, z = spat_df.loc[['k', 'y', 'z']].values
    # initialize spectral values
    spec_values = np.empty((f_reqs.size, y.size), dtype=float)
    for kval in np.unique(k):  # loop over the unique requested k values
        spec_mask = np.isclose(k, kval)  # cols in spec_values corr. to this kval
        # make sure that kval is in con_tc
        if not np.any(np.isclose(kval, con_tc.loc['k'])):
            if warn_datacon:  # throw warning if requested
                warnings.warn(f'Requested component "{kval}" does not exist in constraints! '
                              + 'Cannot interpolate spectra. Trying to use Kaimal instead.',
                              Warning, stacklevel=2)
            spec_values[:, spec_mask] = kaimal_spectrum(f, spat_df.iloc[:, spec_mask], **kwargs)
            continue
        con_mask = (con_tc.loc['k'] == kval).values
        ypts = con_tc.iloc[2, con_mask].values.astype(float)
        zpts = con_tc.iloc[3, con_mask].values.astype(float)
        k_time = con_tc.iloc[4:, con_mask]
        spc_data = 2 * np.abs(np.fft.rfft(k_time, axis=0) / nt)**2 / df
        freq = np.arange(spc_data.shape[0]) * df
        for i_f, f_req in enumerate(f_reqs):
            f_idx = np.argmax(np.isclose(f_req, freq))  # idx of req'd f in con_tc arr
            vals = spc_data[f_idx, :]  # pull out that frequency/component
            spec_values[i_f, spec_mask] = interpolator((ypts, zpts), vals,
                                                       (y[spec_mask], z[spec_mask]))
    return spec_values


def kaimal_spectrum(f, spat_df, u_ref=_DEF_KWARGS['u_ref'], **kwargs):
    """Kaimal PSD as specified in IEC 61400-1 Ed. 3.
    f is (nf,); k, y and z are (n_sp,), u_ref is float or int. returns (nf, n_sp,).
    No std scaling -- that's done with the magnitudes.

    Parameters
    ----------
    f : array-like
        [Hz] Frequency(s) for which PSD is to be calculated. Size is ``n_f``.
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    u_ref : int/float, optional
        [m/s] Mean wind speed at reference height.
    **kwargs
        Unused (optional) keyword arguments.

    Returns
    -------
    spec_values : np.array
        [m^2/s] PSD values for the given spatial locations(s)/component(s).
        Dimension is ``(n_f, n_sp,)``.
    """
    kwargs = {**{'u_ref': u_ref}, **kwargs}  # add dflts if not given
    k, y, z = spat_df.loc[['k', 'y', 'z']].values  # in case pd.series passed in
    f = np.reshape(f, (-1, 1))  # convert to column array
    lambda_1 = 0.7 * z * (z < 60) + 42 * (z >= 60)  # length scale changes with z
    l_k = lambda_1 * (8.1 * np.isclose(k, 0) + 2.7 * np.isclose(k, 1) + 0.66 * np.isclose(k, 2))
    tau = np.reshape((l_k / kwargs['u_ref']), (1, -1))  # L_k / U. row vector
    spec_values = (4 * tau) / np.power(1. + 6 * tau * f, 5. / 3.)  # Kaimal 1972
    return spec_values.astype(float)  # pandas causes object issues, ensure float
