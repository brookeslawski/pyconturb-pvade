# -*- coding: utf-8 -*-
"""Define how the mean wind speed varies with y and z.

For example, the spatial variation of U could be a power law by height, which
is the wind profile specified in IEC 61400-1 Eds. 3 and 4. A user can also specify a
custom function to model the wind speed variation as a function of ``y`` and
``z``. See the notes in ``get_wsp_values`` below for more details.
"""
import warnings

import numpy as np

from pyconturb._utils import _DEF_KWARGS, interpolator


def get_wsp_values(spat_df, wsp_func, **kwargs):
    """Mean wind speed for points/components in ``spat_df``.

    The ``wsp_func`` must be a function of the form::

        wsp_values = wsp_func(spat_df, **kwargs)

    where ``spat_df`` is the spatial dataframe. You can use the
    profile functions built into PyConTurb (see below) or define your own
    custom functions. The output is assumed to be in m/s.

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    wsp_func : function
        Function to map y and z to a wind speed in m/s.
    **kwargs
        Keyword arguments to pass into ``wsp_func``.

    Returns
    -------
    wsp_values : np.array
        [m/s] Mean wind speeds for the given spatial locations(s)/component(s).
        Dimension is ``(n_sp,)``.
    """
    wsp_values = wsp_func(spat_df, **kwargs)
    # wsp_values = wsp_func(spat_df.loc['y'], spat_df.loc['z'],
    #                       **kwargs) * (spat_df.loc['k'] == 0)
    return np.array(wsp_values)  # convert to array in case series given


def constant_profile(spat_df, u_ref=0, **kwargs):
    """Constant (or zero) mean wind speed on longitudinal turbulence components.

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    u_ref : int/float, optional
        [m/s] Mean wind speed at all locations.
    **kwargs
        Unused (optional) keyword arguments.

    Returns
    -------
    wsp_values : np.array
        [m/s] Mean wind speeds at the specified location(s).
    """
    kwargs = {**{'u_ref': u_ref}, **kwargs}  # if not given, add to kwargs
    wsp_vals = np.ones_like(spat_df.loc['y']) * kwargs['u_ref']  # u, v, and w
    return wsp_vals * (spat_df.loc['k'] == 0)  # v, w set to zero


def data_profile(spat_df, con_tc=None, warn_datacon=True, **kwargs):
    """Mean wind speed interpolated from a TimeConstraint object.

    See the Examples and/or Reference Guide for details on the interpolator logic or for
    how to construct a TimeConstraint object.

    Note! If a component is requested for which there is no constraint, then this
    function will return power profile for u and 0 for v and w. Use the `warn_datacon`
    option to disable the warning about this.

    Parameters
    ----------
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
    wsp_values : np.array
        [m/s] Mean wind speed(s) at the specified location(s).
    """
    if con_tc is None:
        raise ValueError('No data provided!')
    mean_wsp = np.empty(spat_df.shape[1])  # initialize array of mean wind speeds
    for ic, c in enumerate('uvw'):
        mask = np.isclose(con_tc.loc['k'], ic)  # cols con_tc that correspond to u
        mask_sdf = np.isclose(spat_df.loc['k'], ic)  # cols spat_df that correspond to u
        comp_spat_df = spat_df.iloc[:, mask_sdf]  # set of spat_df that is component
        if not sum(mask):
            if warn_datacon:  # throw warning if requested
                warnings.warn(f'{c}-wind does not exist in constraints! '
                              + 'Cannot interpolate mean profile. Trying to use power law.',
                              Warning, stacklevel=2)
            if ic:  # v, w: zero
                mean_wsp[mask_sdf] = constant_profile(comp_spat_df)
            else:  # u: power profile
                mean_wsp[mask_sdf] = power_profile(comp_spat_df, **kwargs)
            continue
        # yp, zp, and vals to interpolate from
        ypts = con_tc.filter(regex=c + '_').loc['y'].values.astype(float)
        zpts = con_tc.filter(regex=c + '_').loc['z'].values.astype(float)
        vals = con_tc.get_time().filter(regex=c + '_').mean().values.astype(float)
        # values to interpolate to
        y, z = comp_spat_df.loc[['y', 'z']].values
        mean_wsp[mask_sdf] = interpolator((ypts, zpts), vals, (y, z))
    return mean_wsp


def power_profile(spat_df, u_ref=_DEF_KWARGS['u_ref'], z_ref=_DEF_KWARGS['z_ref'],
                  alpha=_DEF_KWARGS['alpha'], **kwargs):
    """Power-law profile with height.

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    u_ref : int/float, optional
        [m/s] Mean wind speed at reference height.
    z_ref : int/float, optional
        [m] Reference height.
    alpha : int/float, optional
        [-] Coefficient for the power law.
    **kwargs
        Unused (optional) keyword arguments.

    Returns
    -------
    wsp_values : np.array
        [m/s] Mean wind speed(s) at the specified location(s).
    """
    kwargs = {**{'u_ref': u_ref, 'z_ref': z_ref, 'alpha': alpha},
              **kwargs}  # if not given, add defaults to kwargs
    wsp_vals = kwargs['u_ref'] * (spat_df.loc['z'] / kwargs['z_ref']) ** kwargs['alpha']
    return wsp_vals * (spat_df.loc['k'] == 0)
