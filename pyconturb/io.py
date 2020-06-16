"""Input-output for PyConTurb files
"""
import os
import numpy as np
import pandas as pd


_HAWC2_BIN_FMT = '<f'  # HAWC2 binary turbulence datatype


def df_to_h2turb(turb_df, spat_df, path, prefix=''):
    """pyconturb-style turbulence dataframe to binary files for hawc2

    Notes
    -----
    * The turbulence must have been generated on a y-z grid.
    * The naming convention must be 'u_p0', 'v_p0, 'w_p0', 'u_p1', etc.,
       where the point indices proceed vertically along z before horizontally
       along y.
    """
    nx = turb_df.shape[0]  # turbulence dimensions for reshaping
    ny = len(set(spat_df.loc['y'].values))
    nz = len(set(spat_df.loc['z'].values))
    # make and save binary files for all three components
    for c in 'uvw':
        arr = turb_df.filter(regex=f'{c}_', axis=1).values.reshape((nx, ny, nz))
        bin_path = os.path.join(path, f'{prefix}{c}.bin')
        with open(bin_path, 'wb') as bin_fid:
            arr.astype(np.dtype(_HAWC2_BIN_FMT)).tofile(bin_fid)
    return


def h2turb_to_arr(spat_df, path):
    """Raw-load a hawc2 turbulent binary file to numeric array"""
    ny, nz = pd.unique(spat_df.loc['y']).size, pd.unique(spat_df.loc['z']).size
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
        arr = h2turb_to_arr(spat_df, comp_path)
        nx, ny, nz = arr.shape
        comp_df = pd.DataFrame(arr.reshape(nx, ny*nz)).add_prefix(f'{c}_p')
        turb_df = turb_df.join(comp_df, how='outer')
    turb_df = turb_df[[f'{c}_p{i}' for i in range(2) for c in 'uvw']]
    return turb_df
