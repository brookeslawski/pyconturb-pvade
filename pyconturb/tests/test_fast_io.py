# -*- coding: utf-8 -*-
"""Check that running a bts through FAST matches directly-loaded bts
"""
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from pyconturb.io import df_to_bts, bts_to_df
from pyconturb.simulation import gen_turb
from pyconturb._utils import gen_spat_grid



@pytest.mark.openfast
@pytest.mark.skipci  # don't run in CI
def test_binary_thru_fast7():
    """create binary turbulence, run through fast v7, and reload from output
    """
    # turbulence inputs
    z_hub, l_blade = 60.2031, 25  # hub height, blade length
    y = [-2*l_blade, -l_blade, 0, l_blade]  # x-components of turb grid (offset to check loading)
    z = [z_hub - l_blade, z_hub, z_hub + l_blade]  # z-components of turb grid (offset to check loading)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2,
              'z_ref': z_hub, 'T': 20, 'nt': 40}
    coh_model = 'iec'
    spat_df = gen_spat_grid(y, z)
    
    # get the pyconturb index of the hub-height point
    hh_pts = spat_df.loc[:,(spat_df.loc['y'] == 0) & np.isclose(spat_df.loc['z'], z_hub)]
    hh_idx = hh_pts.filter(regex='u_').columns[0].lstrip('u')
    
    # paths, directories, and file names
    test_dir = os.path.dirname(__file__)  # test directory
    sim_dir = os.path.join(test_dir, 'fast_sims')
    exe_path = os.path.join(sim_dir, 'FAST.exe')
    inp_path = os.path.join(sim_dir, 'WP_0.75MW.fst')
    bts_path = os.path.join(sim_dir, 'test.bts')
    out_path = os.path.join(sim_dir, 'WP_0.75MW.out')
    
    if not os.path.isfile(exe_path):
        warnings.warn('***FAST executable not found!!!***')
    
    # 1. generate turbulence files and save to bts
    orig_df = gen_turb(spat_df, coh_model=coh_model,
                        **kwargs)
    df_to_bts(orig_df, spat_df, bts_path)
    
    # 2. run FAST on htc file
    proc = subprocess.run([exe_path, inp_path], cwd=sim_dir, capture_output=True)
    if proc.returncode:
        with open(os.path.join(sim_dir, 'test.err'), 'wb') as f:
            f.write(proc.stderr)
            f.write(proc.stdout)
        raise ValueError('Error running FAST! See test.err')
    
    # 3. load results
    bts_df = bts_to_df(bts_path)  # reload bts
    fast_out = np.loadtxt(out_path, skiprows=8)  # fast output
    fast_df = pd.DataFrame(fast_out[:, 1:4], index=fast_out[:, 0])  # x, y, z
    
    # 4. check results are equal
    for i, c in enumerate('uvw'):
        np.testing.assert_allclose(orig_df[c + hh_idx], bts_df[c + hh_idx], rtol=np.inf,
                                    atol=1e-5)
        # pd.testing.assert_series_equal(orig_df[f'{c}_p4'], bts_df[f'{c}_p4'])
        np.testing.assert_allclose(orig_df[c + hh_idx].iloc[1:],
                                    fast_df[i].iloc[:-1], rtol=np.inf, atol=1e-5)
    
    # 5. clean up
    os.remove(out_path)
    os.remove(bts_path)


if __name__ == '__main__':
    test_binary_thru_fast7()
