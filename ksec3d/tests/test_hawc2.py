# -*- coding: utf-8 -*-
"""Test functions for converting to hawc2 binary and reading into hawc2

Author
------
Jenni Rinker
rink@dtu.dk
"""
import os
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from ksec3d.core.simulation import gen_turb
from ksec3d.core.helpers import gen_spat_grid, make_hawc2_input, df_to_hawc2
from ksec3d.io.hawc2 import dat_to_df


def test_binary_thru_hawc2():
    """create binary turbulence, run through hawc2, and reload from h2 output
    """
    # turbulence inputs
    z_hub, l_blade = 119, 90  # hub height, blade length
    y = [-l_blade, l_blade]  # x-components of turb grid
    z = [z_hub - l_blade, z_hub, z_hub + l_blade]  # z-components of turb grid
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2,
              'z_hub': z_hub, 'T': 50, 'dt': 1.}
    coh_model, spc_model, wsp_profile = 'iec', 'kaimal', 'none'
    spat_df = gen_spat_grid(y, z)

    # paths, directories, and file names
    test_dir = os.getcwd()  # test directory
    testdata_dir = os.path.join(test_dir, 'data')  # data directory
    tmp_dir = os.path.join(test_dir, 'tmp')  # temporary directory for test
    htc_name = 'load_save_turb.htc'  # hawc2 simulation template
    htc_path = os.path.join(testdata_dir, htc_name)  # path to htc template
    new_htc_path = os.path.join(tmp_dir, htc_name)  # htc file in tmp/
    csv_path = os.path.join(tmp_dir, 'turb_df.csv')  # save pandas turb here
    bat_path = os.path.join(tmp_dir, 'run_hawc2.bat')  # bat file to run h2
    hawc2_exe = 'C:/Users/rink/Documents/hawc2/HAWC2_all_12-4/HAWC2MB.exe'

    if not os.path.isfile(hawc2_exe):
        warnings.warn('***HAWC2 executable not found!!! Skipping test.***')
        return

    # 1. create temp directory
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    # 2. copy htc file there, replacing values
    T, dt, wsp = kwargs['T'], kwargs['dt'], kwargs['v_hub']  # needed in htc
    str_cntr_pos0, str_mann, str_output = make_hawc2_input(tmp_dir,
                                                           spat_df, **kwargs)
    with open(htc_path, 'r') as old_fid:
        with open(new_htc_path, 'w') as new_fid:
            for line in old_fid:
                new_line = eval('f\'' + line.rstrip() + '\'') + '\n'
                new_fid.write(new_line)

    # 4. generate turbulence files
    turb_df = gen_turb(spat_df,
                       coh_model=coh_model, spc_model=spc_model,
                       wsp_profile=wsp_profile, scale=True,
                       seed=False, **kwargs)
    df_to_hawc2(turb_df, spat_df, tmp_dir)
    turb_df.reset_index().to_csv(csv_path)
    del turb_df

    # 3. run HAWC2 on htc file
    with open(bat_path, 'w') as bat_fid:
        bat_fid.write(f'cd {tmp_dir}\n' +
                      f'"{hawc2_exe}" {htc_name}')
    subprocess.call(f'{bat_path}', shell=True)

    # 4. load results
    turb_df = pd.read_csv(csv_path).set_index('index')  # simulated results
    dat_df = dat_to_df(new_htc_path).set_index('time')  # hawc2 results

    # 5. compare results
    time_vec = np.arange(4, 10)
    turb_tuples = [('vxt_p0', -1, 'vyg_p0', 1),  # u
                   ('vyt_p0', -1, 'vxg_p0', 1),  # v
                   ('vzt_p0', 1, 'vzg_p0', -1)]  # w
    for py_key, py_sign, h2_key, h2_sign in turb_tuples:
        py_turb = np.interp(time_vec, turb_df.index, py_sign * turb_df[py_key])
        h2_turb = np.interp(time_vec, dat_df.index, h2_sign * dat_df[h2_key])
        np.testing.assert_allclose(py_turb, h2_turb, atol=1e-3)

    # 6. delete temp directory
    shutil.rmtree(tmp_dir)
