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


_spat_colnames = ['k', 'p_id', 'x', 'y', 'z']


def combine_spat_df(left_df, right_df,
                    drop_duplicates=True):
    """combine two spatial dataframes, changing point index of right_df
    """
    if left_df.size == 0:
        return right_df.copy()
    if right_df.size == 0:
        return left_df.copy()
    left_df = left_df.copy()  # don't want to overwrite original dataframes
    right_df = right_df.copy()

    max_left_pid = int(left_df[['p_id']].applymap(lambda s: int(s[1:])).max())
    right_df['p_id'] = right_df[['p_id']]\
        .applymap(lambda s: f'p{int(s[1:])+max_left_pid+1}')
    comb_df = pd.concat((left_df, right_df), axis=0)
    if drop_duplicates:
        comb_df = comb_df.drop_duplicates(subset=['k', 'x', 'y', 'z'])
    comb_df = comb_df.reset_index(drop=True)
    return comb_df


def complex_interp(df, **kwargs):
    """linearly interpolate NaNs in a complex dataframe (pandas method broken)
    """
    re = pd.DataFrame(index=df.index, columns=df.columns)
    imag = pd.DataFrame(index=df.index, columns=df.columns)
    re[:] = np.real(df)
    imag[:] = np.imag(df)
    re.iloc[np.isnan(df).values] = np.nan
    imag.iloc[np.isnan(df).values] = np.nan
    return re.interpolate(**kwargs) + 1j * imag.interpolate(**kwargs)


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


def get_iec_sigk(spat_df, **kwargs):
    """get sig_k for iec
    """
    sig = kwargs['i_ref'] * (0.75 * kwargs['v_hub'] + 5.6)  # std dev
    sig_k = sig * (1.0 * (spat_df.k == 'vxt') + 0.8 * (spat_df.k == 'vyt') +
                   0.5 * (spat_df.k == 'vzt')).values
    return sig_k


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
                           columns=_spat_colnames)
    spat_df[['x', 'y', 'z']] = spat_df[['x', 'y', 'z']].astype(float)
    return spat_df


def h2t_to_uvw(turb_df):
    """convert turbulence dataframe with hawc2 turbulence coord sys to uvw
    """
    comp_tups = [('vxt', 'u', -1), ('vyt', 'v', -1), ('vzt', 'w', 1)]
    new_turb_df = pd.DataFrame(index=turb_df.index)
    for comp_h2t, comp_uvw, sign in comp_tups:
        old_cols = [s for s in turb_df.columns if comp_h2t in s]
        new_cols = [s.replace(comp_h2t, comp_uvw) for s in old_cols]
        new_turb_df[new_cols] = sign * turb_df.loc[:, old_cols]

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


def spat_to_pair_df(spat_df):
    """convert spat_df to pair_df
    """
    n_s = spat_df.shape[0]  # no. of spatial points
    if n_s == 1:
        n_pairs = 0
        return pd.DataFrame(np.empty((n_pairs, 8)),
                            columns=['k1', 'x1', 'y1', 'z1', 'k2', 'x2',
                                     'y2', 'z2'])  # df input to coherence fcn
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


def spc_to_mag(spc_df, spat_df, df, n_t, **kwargs):
    """Convert spectral dataframe to magnitudes
    """
    mags_df = np.sqrt(spc_df * df / 2)
    mags_df.iloc[0, :] = 0.  # set dc component to zero

    if kwargs['scale']:
        sum_magsq = 2 * (mags_df ** 2).sum(axis=0).values.reshape(1, -1)
        sig_k = get_iec_sigk(spat_df, **kwargs).reshape(1, -1)
        alpha = np.sqrt((n_t - 1) / n_t
                        * (sig_k ** 2) / sum_magsq)  # scaling factor
    else:
        alpha = 1
    mags_df = alpha * mags_df

    return mags_df
