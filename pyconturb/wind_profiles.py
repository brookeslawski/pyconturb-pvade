# -*- coding: utf-8 -*-
"""Define how the mean wind speed varies with y and z.

For example, the spatial variation of U could be a power law by height, which
is the wind profile specified in IEC 61400-1 Ed. 3. A user can also specify a
custom function to model the wind speed variation as a function of ``y`` and
``z``. See the notes in ``get_wsp_values`` below for more details.
"""
import numpy as np

from pyconturb._utils import _DEF_KWARGS


def get_wsp_values(spat_df, wsp_func, **kwargs):
    """Mean wind speed for points/components in ``spat_df``.

    The ``wsp_func`` must be a function of the form::

        wsp_values = wsp_func(y, z, **kwargs)

    where y and z can be floats, np.arrays or pandas.Series. You can use the
    profile functions built into PyConTurb (see below) or define your own
    custom functions. The output is assumed to be in m/s.

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, p_id, x, y, z]``, and each of the ``n_sp`` rows corresponds
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
    wsp_values = wsp_func(spat_df.y, spat_df.z, **kwargs) * (spat_df.k == 0)
    return np.array(wsp_values)  # convert to array in case series given


def constant_profile(y, z, u_ref=0, **kwargs):
    """Constant (or zero) mean wind speed.

    Parameters
    ----------
    y : array-like
        [m] Location of point(s) in the lateral direction. Can be int/float,
        np.array or pandas.Series.
    z : array-like
        [m] Location of point(s) in the vertical direction. Can be int/float,
        np.array or pandas.Series.
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
    return np.ones_like(y) * kwargs['u_ref']


def power_profile(y, z, u_ref=_DEF_KWARGS['u_ref'], z_ref=_DEF_KWARGS['z_ref'],
                  alpha=_DEF_KWARGS['alpha'], **kwargs):
    """Power-law profile with height.

    Parameters
    ----------
    y : array-like
        [m] Location of point(s) in the lateral direction. Can be int/float,
        np.array or pandas.Series.
    z : array-like
        [m] Location of point(s) in the vertical direction. Can be int/float,
        np.array or pandas.Series.
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
    return kwargs['u_ref'] * (z / kwargs['z_ref']) ** kwargs['alpha']
