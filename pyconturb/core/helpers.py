# -*- coding: utf-8 -*-
"""Miscellaneous helper functions

Author
------
Jenni Rinker
rink@dtu.dk
"""
import os

import numpy as np
import pandas as pd


_spat_colnames = ['k', 'p_id', 'x', 'y', 'z']  # column names of spatial df
_HAWC2_BIN_FMT = '<f'  # HAWC2 binary turbulence datatype
_HAWC2_TURB_COOR = {'u': -1, 'v': -1, 'w': 1}  # hawc2 turb xyz to uvw


def combine_spat_df(top_df, bot_df, drop_duplicates=True):
    """combine two spatial dataframes, changing point index of right_df
    """
    top_df, bot_df = top_df.copy(), bot_df.copy()  # no change original when update p_id
    top_df.p_id -= top_df.p_id.min()  # p_id must start at zero
    bot_df.p_id -= bot_df.p_id.min()
    if top_df.size:  # if non-empty top_df, add p_id of top to bot_df
        bot_df.p_id += top_df.p_id.max() + 1
    comb_df = pd.concat((top_df, bot_df))
    if drop_duplicates:
        comb_df = comb_df.drop_duplicates(subset=['k', 'x', 'y', 'z'])
    comb_df = comb_df.reset_index(drop=True)
    return comb_df


def df_to_h2turb(turb_df, spat_df, path, prefix=''):
    """ksec3d-style turbulence dataframe to binary files for hawc2

    Notes
    -----
    * The turbulence must have been generated on a y-z grid.
    * The naming convention must be 'u_p0', 'v_p0, 'w_p0', 'u_p1', etc.,
       where the point indices proceed vertically along z before horizontally
       along y.
    """
    nx = turb_df.shape[0]  # turbulence dimensions for reshaping
    ny = len(set(spat_df.y.values))
    nz = len(set(spat_df.z.values))
    # make and save binary files for all three components
    for c in 'uvw':
        coeff = _HAWC2_TURB_COOR[c]
        arr = coeff * turb_df.filter(regex=f'{c}_', axis=1).values.reshape((nx, ny, nz))
        bin_path = os.path.join(path, f'{prefix}{c}.bin')
        with open(bin_path, 'wb') as bin_fid:
            arr.astype(np.dtype(_HAWC2_BIN_FMT)).tofile(bin_fid)
    return


def get_iec_sigk(spat_df, **kwargs):
    """get sig_k for iec
    """
    sig = kwargs['i_ref'] * (0.75 * kwargs['v_hub'] + 5.6)  # std dev
    sig_k = sig * (1.0 * (spat_df.k == 0) + 0.8 * (spat_df.k == 1) +
                   0.5 * (spat_df.k == 2)).values
    return sig_k


def gen_spat_grid(y, z, comps=[0, 1, 2]):
    """Generate spat_df (all turbulent components and grid defined by x and z)

    Notes
    -----
    0=u is downwind, 2=w is vertical and 1=v is lateral (right-handed
    coordinate system).
    """
    ys, zs = np.meshgrid(y, z)  # make a meshgrid
    ks = np.array(comps)  # sanitizing
    xs = np.zeros_like(ys)  # all points in a plane
    ps = np.arange(xs.size)  # point indices
    spat_arr = np.c_[np.tile(comps, xs.size),
                     np.repeat(np.c_[ps, xs.T.ravel(), ys.T.ravel(), zs.T.ravel()],
                               ks.size, axis=0)]  # create array using numpy
    spat_df = pd.DataFrame(spat_arr, columns=_spat_colnames)  # create dataframe
    return spat_df


def h2turb_to_arr(spat_df, path):
    """raw-load a hawc2 turbulent binary file to numeric array"""
    ny, nz = pd.unique(spat_df.y).size, pd.unique(spat_df.z).size
    bin_arr = np.fromfile(path, dtype=np.dtype(_HAWC2_BIN_FMT))
    nx = bin_arr.size // (ny * nz)
    if (nx * ny * nz) != bin_arr.size:
        raise ValueError('Binary file size does not match spat_df!')
    bin_arr.shape = (nx, ny, nz)
    return bin_arr


def h2turb_to_df(spat_df, path, prefix=''):
    """load a hawc2 binary file into a pandas datafram with transform to uvw"""
    turb_df = pd.DataFrame()
    for c in 'uvw':
        comp_path = os.path.join(path, f'{prefix}{c}.bin')
        arr = _HAWC2_TURB_COOR[c] * h2turb_to_arr(spat_df, comp_path)
        nx, ny, nz = arr.shape
        comp_df = pd.DataFrame(arr.reshape(nx, ny*nz)).add_prefix(f'{c}_p')
        turb_df = turb_df.join(comp_df, how='outer')
    turb_df = turb_df[[f'{c}_p{i}' for i in range(2) for c in 'uvw']]
    return turb_df


def h2t_to_uvw(turb_df):
    """convert turbulence dataframe with hawc2 turbulence coord sys to uvw
    """
    new_turb_df = pd.DataFrame(index=turb_df.index)
    h2_comps = ['vxt', 'vyt', 'vzt']
    for ic, c in enumerate('uvw'):
        new_cols = [f'{c}_p{i}' for i in range(turb_df.shape[1] // 3)]
        new_turb_df[new_cols] = _HAWC2_TURB_COOR[c] \
            * turb_df.filter(regex=f'{h2_comps[ic]}_', axis=1)
    return new_turb_df


def make_hawc2_input(turb_dir, spat_df, **kwargs):
    """return strings for the hawc2 input files
    """
    # string of center position
    z_hub = kwargs['z_hub']
    str_cntr_pos0 = '  center_pos0             0.0 0.0 ' + \
        f'{-z_hub:.1f} ; hub height\n'

    # string for mann model block
    T, dt = kwargs['T'], kwargs['dt']
    y, z = set(spat_df.y.values), set(spat_df.z.values)
    n_x, du = int(np.ceil(T / dt)), dt * kwargs['v_hub']
    n_y, dv = len(y), (max(y) - min(y)) / (len(y) - 1)
    n_z, dw = len(z), (max(z) - min(z)) / (len(z) - 1)
    str_mann = '  begin mann ;\n' + \
               f'    filename_u {turb_dir}/u.bin ; \n' + \
               f'    filename_v {turb_dir}/v.bin ; \n' + \
               f'    filename_w {turb_dir}/w.bin ; \n' + \
               f'    box_dim_u {n_x:.0f} {du:.1f} ; \n' + \
               f'    box_dim_v {n_y:.0f} {dv:.1f} ; \n' + \
               f'    box_dim_w {n_z:.0f} {dw:.1f} ; \n' + \
               f'    dont_scale 1 ; \n' + \
               '  end mann '

    # string for output
    pts_df = spat_df.loc[spat_df.k == 'vxt', ['x', 'y', 'z']]
    str_output = ''
    for i_p in pts_df.index:
        x_p = -pts_df.loc[i_p, 'y']
        y_p = -pts_df.loc[i_p, 'x']
        z_p = -pts_df.loc[i_p, 'z']
        str_output += f'  wind free_wind 1 {x_p:.1f} {y_p:.1f} {z_p:.1f}' + \
            f' # wind_p{i_p//3} ; \n'

    return str_cntr_pos0, str_mann, str_output


def rotate_time_series(ux, uy, uz):
    """Yaw and pitch time series so v- and w-directions have zero mean

        Args:
            ux (numpy array): array of x-sonic velocity
            uy (numpy array): array of y-sonic velocity
            uz (numpy array): array of z-sonic velocity

        Returns:
            x_rot (numpy array): [n_t x 3] array of rotated data (yaw+pitch)
            x_yaw (numpy array): [n_t x 3] array of rotated data (yaw)
    """

    # return all NaNs if any component is all nan values
    if all(np.isnan(ux))*all(np.isnan(uy))*all(np.isnan(uz)):
        u = np.zeros(ux.shape)
        v = np.zeros(uy.shape)
        w = np.zeros(uz.shape)
        u[:] = np.nan
        v[:] = np.nan
        w[:] = np.nan

    # if at least one data point in all three components
    else:

        # combine velocities into array
        x_raw = np.concatenate((ux.reshape(ux.size, 1),
                                uy.reshape(ux.size, 1),
                                uz.reshape(ux.size, 1)), axis=1)

        # interpolate out any NaN values
        for i_comp in range(x_raw.shape[1]):
            x = x_raw[:, i_comp]
            idcs_all = np.arange(x.size)
            idcs_notnan = np.logical_not(np.isnan(x))
            x_raw[:, i_comp] = np.interp(idcs_all,
                                         idcs_all[idcs_notnan], x[idcs_notnan])

        # rotate through yaw angle
        theta = np.arctan(np.nanmean(x_raw[:, 1]) / np.nanmean(x_raw[:, 0]))
        A_yaw = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
        x_yaw = x_raw @ A_yaw

        # rotate through pitch angle
        phi = np.arctan(np.nanmean(x_yaw[:, 2]) / np.nanmean(x_yaw[:, 0]))
        A_pitch = np.array([[np.cos(phi), 0, -np.sin(phi)],
                            [0, 1, 0],
                            [np.sin(phi), 0, np.cos(phi)]])
        x_rot = x_yaw @ A_pitch

        # define rotated velocities
        u = x_rot[:, 0]
        v = x_rot[:, 1]
        w = x_rot[:, 2]

    return u, v, w


def spc_to_mag(spc_np, spat_df, df, n_t, **kwargs):
    """Convert spectral dataframe to magnitudes
    """
    if 'scale' not in kwargs.keys():
        raise ValueError('Missing keyword argument "scale"!')
    spc_np = spc_np.astype(float)
    mags_np = np.sqrt(spc_np * df / 2)
    mags_np[0, :] = 0.  # set dc component to zero

    if kwargs['scale']:
        sum_magsq = 2 * (mags_np ** 2).sum(axis=0).reshape(1, -1)
        sig_k = get_iec_sigk(spat_df, **kwargs).reshape(1, -1)
        alpha = np.sqrt((n_t - 1) / n_t
                        * (sig_k ** 2) / sum_magsq)  # scaling factor
    else:
        alpha = 1
    mags_np = alpha * mags_np

    return mags_np
