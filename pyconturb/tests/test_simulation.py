# -*- coding: utf-8 -*-
"""Test functions in simulation.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd

from pyconturb import gen_turb, TimeConstraint
from pyconturb.sig_models import iec_sig
from pyconturb.spectral_models import kaimal_spectrum
from pyconturb.wind_profiles import constant_profile, power_profile
from pyconturb._utils import gen_spat_grid, _spat_rownames


def test_iec_turb_mn_std_dev():
    """test that iec turbulence has correct mean and std deviation"""
    # given
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70, 'T': 300, 'dt': 1}
    sig_theo = np.array([1.834, 1.4672, 0.917, 1.834, 1.4672, 0.917])
    u_theo = np.array([10, 0, 0, 10.27066087, 0, 0])
    # when
    turb_df = gen_turb(spat_df, **kwargs)
    # then
    np.testing.assert_allclose(sig_theo, turb_df.std(axis=0), atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, turb_df.mean(axis=0),  atol=0.01)


def test_gen_turb_con():
    """mean & std of iec turbulence, con'd turb is regen'd, correct columns
    """
    # given -- constraining points
    con_spat_df = pd.DataFrame([[0, 0, 0, 70]], columns=_spat_rownames)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70, 'T': 300,
              'dt': 0.5, 'seed': 1337}
    coh_model = 'iec'
    con_turb_df = gen_turb(con_spat_df.T, coh_model=coh_model, **kwargs)
    con_tc = TimeConstraint().from_con_data(con_spat_df=con_spat_df, con_turb_df=con_turb_df)
    # given -- simulated, constrainted turbulence
    y, z = 0, [70, 72]
    spat_df = gen_spat_grid(y, z)
    wsp_func, sig_func, spec_func = power_profile, iec_sig, kaimal_spectrum
    sig_theo = np.tile([1.834, 1.4672, 0.917], 2)  # sig_u, sig_v, sig_w
    u_theo = np.array([10, 0, 0, 10.05650077210035, 0, 0])  # U1, ... U2, ...
    theo_cols = [f'{"uvw"[ic]}_p{ip}' for ip in range(2)for ic in range(3)]
    # when
    sim_turb_df = gen_turb(spat_df, con_tc=con_tc, wsp_func=wsp_func, sig_func=sig_func,
                           spec_func=spec_func, coh_model=coh_model, **kwargs)
    # then (std dev, mean, and regen'd time series should be close; right colnames)
    pd.testing.assert_index_equal(sim_turb_df.columns, pd.Index(theo_cols))
    np.testing.assert_allclose(sig_theo, sim_turb_df.std(axis=0), atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, sim_turb_df.mean(axis=0), atol=0.01)
    np.testing.assert_allclose(con_turb_df.u_p0, sim_turb_df.u_p0, atol=0.01)


if __name__ == '__main__':
    test_iec_turb_mn_std_dev()
    test_gen_turb_con()
