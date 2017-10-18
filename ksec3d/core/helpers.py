# -*- coding: utf-8 -*-
"""Miscellaneous helper functions

Author
------
Jenni Rinker
rink@dtu.dk
"""
import itertools
import os

import numpy as np
import pandas as pd


def gen_spat_grid(y, z):
    """Generate spat_df (all turbulent components and grid defined by x and z)

    Notes
    -----
    This coordinate system is aligned with HAWC2's turbulence coordinate
    system. In other words, x is directed upwind, z is vertical, and y is
    lateral according to a right-hand coordinate system.
    """
    ys, zs = np.meshgrid(y, z)
    ks = np.array(['vxt', 'vyt', 'vzt'])
    xs = np.zeros_like(ys)
    ps = [f'p{i:.0f}' for i in np.arange(xs.size)]
    spat_arr = np.vstack((np.tile(ks, xs.size),
                          np.repeat(ps, ks.size),
                          np.repeat(xs.T.reshape(-1), ks.size),
                          np.repeat(ys.T.reshape(-1), ks.size),
                          np.repeat(zs.T.reshape(-1), ks.size))).T
    spat_df = pd.DataFrame(spat_arr,
                           columns=['k', 'p_id', 'x', 'y', 'z'])
    spat_df[['x', 'y', 'z']] = spat_df[['x', 'y', 'z']].astype(float)
    return spat_df


def get_iec_sigk(spat_df, **kwargs):
    """get sig_k for iec
    """
    sig = kwargs['i_ref'] * (0.75 * kwargs['v_hub'] + 5.6)  # std dev
    sig_k = sig * (1.0 * (spat_df.k == 'vxt') + 0.8 * (spat_df.k == 'vyt') +
                   0.5 * (spat_df.k == 'vzt')).values
    return sig_k


def df_to_hawc2(turb_df, spat_df, path):
    """ksec3d-style turbulence dataframe to binary files for hawc2

    Notes
    -----
    1. The turbulence must have been generated on a grid.
    2. The naming convention must be 'u_p0', 'v_p0, 'w_p0', 'u_p1', etc.,
       where the point indices proceed vertically along z before horizontally
       along y.
    3. The turbulence boxes should have no mean wind speed profiles added.
    4. The turbulence df must be defined according to HAWC2's turbulence
       (x, y, z) coordinate system. Thus, the conversion to the (u, v, w)
       binary coordinate system (bin) from the hawc2 coordinate system is as
       follows:
           u(bin) = -u(hawc2)
           v(bin) = -v(hawc2)
           w(bin) =  w(hawc2)
    """
    # define dimensions
    n_x = turb_df.shape[0]
    n_y = len(set(spat_df.y.values))
    n_z = len(set(spat_df.z.values))

    # convert to hawc2 coordinate systems
    u_bin = -turb_df[[s for s in turb_df.columns
                      if 'vxt_' in s]].values.reshape((n_x, n_y, n_z))
    v_bin = -turb_df[[s for s in turb_df.columns
                      if 'vyt_' in s]].values.reshape((n_x, n_y, n_z))
    w_bin = turb_df[[s for s in turb_df.columns
                     if 'vzt_' in s]].values.reshape((n_x, n_y, n_z))

    # save binary files
    for comp, turb in zip(['u', 'v', 'w'], [u_bin, v_bin, w_bin]):
        bin_path = os.path.join(path, f'{comp}.bin')
        with open(bin_path, 'wb') as bin_fid:
            turb.astype(np.dtype('<f')).tofile(bin_fid)

    return


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


def spat_to_pair_df(spat_df):
    """convert spat_df to pair_df
    """
    n_s = spat_df.shape[0]  # no. of spatial points
    n_pairs = int(np.math.factorial(n_s) / 2 /
                  np.math.factorial(n_s - 2))  # no. of combos
    pair_df = pd.DataFrame(np.empty((n_pairs, 8)),
                           columns=['k1', 'x1', 'y1', 'z1', 'k2', 'x2',
                                    'y2', 'z2'])  # df input to coherence fcn

    i_df = 0  # initialize counter
    ii, jj = [], []  # use these index vectors later during cholesky decomp
    for (i, j) in itertools.combinations(spat_df.index, 2):
        pair_df.loc[i_df, ['k1', 'x1', 'y1', 'z1']] = \
            spat_df.loc[i, ['k', 'x', 'y', 'z']].values
        pair_df.loc[i_df, ['k2', 'x2', 'y2', 'z2']] = \
            spat_df.loc[j, ['k', 'x', 'y', 'z']].values
        i_df += 1
        ii.append(i)  # save index
        jj.append(j)  # save index

    return pair_df
