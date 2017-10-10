# -*- coding: utf-8 -*-
"""Test functions in coherence.py
"""

import numpy as np
import pandas as pd

from ksec3d.core.simulation import get_phasors, gen_turb
from ksec3d.core.helpers import gen_spat_grid


def test_get_phases():
    """Check the value for get_phases
    """
    # given
    spat_df = pd.DataFrame([['u', 0, 0, 50],
                            ['u', 0, 0, 51]],
                           columns=['k', 'x', 'y', 'z'])
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2}
    coh_model, T, dt, seed = 'iec', 8, 4, None
    coh_theory = 0.8606565849
    tol = 0.05  # stochastic tolerance
    n_real = 100  # no. realizations for ensemble averaging

    # when
    coh = 0
    den1 = 0
    den2 = 0
    for i_real in range(n_real):
        phasors = get_phasors(spat_df,
                              coh_model=coh_model, T=T, dt=dt,
                              seed=seed,
                              **kwargs)
        Xi, Xj = phasors.iloc[0, 1], phasors.iloc[1, 1]
        coh += Xi * np.conj(Xj)
        den1 += Xi * np.conj(Xi)
        den2 += Xj * np.conj(Xj)
    corr = coh / np.sqrt(den1 * den2)

    # then
    assert (corr < coh_theory*(1 + tol)) & (corr > coh_theory*(1 - tol))


def test_iec_turb_std_dev():
    """test that iec turbulence has correct generated std deviation
    """
    x, z = 0, 90  # given
    spat_df = gen_spat_grid(x, z)
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': z}
    coh_model, T, dt = 'iec', 8, 4
    sig_theo = np.array([1.834, 1.4672, 0.917])
    turb_df = gen_turb(spat_df,
                       coh_model=coh_model, spc_model='kaimal', T=T, dt=dt,
                       **kwargs)  # when
    np.testing.assert_allclose(sig_theo, turb_df.std(axis=0),
                               rtol=1e-3)
