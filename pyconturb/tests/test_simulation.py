# -*- coding: utf-8 -*-
"""Test functions in simulation.py

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import pandas as pd
import pytest

from pyconturb import gen_turb, TimeConstraint
from pyconturb.coherence import generate_coherence_file
from pyconturb.sig_models import iec_sig
from pyconturb.spectral_models import kaimal_spectrum
from pyconturb.wind_profiles import power_profile
from pyconturb._utils import gen_spat_grid, _spat_rownames, get_freq


def test_iec_turb_mn_std_dev():
    """test that iec turbulence has correct mean and std deviation"""
    # given
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70, 'T': 300, 'nt': 300}
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
    con_spat_df = pd.DataFrame([[0, 0, 0, 70]], columns=_spat_rownames).T
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70, 'T': 300,
              'nt': 600, 'seed': 1337, 'coh_model': 'iec'}
    con_turb_df = gen_turb(con_spat_df, **kwargs)
    con_tc = TimeConstraint().from_con_data(con_spat_df=con_spat_df.T,
                                            con_turb_df=con_turb_df)  # old spat_df was T
    # given -- simulated, constrainted turbulence
    y, z = 0, [70, 72]
    spat_df = gen_spat_grid(y, z)
    wsp_func, sig_func, spec_func = power_profile, iec_sig, kaimal_spectrum
    sig_theo = np.tile([1.834, 1.4672, 0.917], 2)  # sig_u, sig_v, sig_w
    u_theo = np.array([10, 0, 0, 10.05650077210035, 0, 0])  # U1, ... U2, ...
    theo_cols = [f'{"uvw"[ic]}_p{ip}' for ip in range(2)for ic in range(3)]
    # when
    sim_turb_df = gen_turb(spat_df, con_tc=con_tc, wsp_func=wsp_func, sig_func=sig_func,
                           spec_func=spec_func, **kwargs)
    # then (std dev, mean, and regen'd time series should be close; right colnames)
    pd.testing.assert_index_equal(sim_turb_df.columns, pd.Index(theo_cols))
    np.testing.assert_allclose(sig_theo, sim_turb_df.std(axis=0), atol=0.01, rtol=0.50)
    np.testing.assert_allclose(u_theo, sim_turb_df.mean(axis=0), atol=0.01)
    np.testing.assert_allclose(con_turb_df.u_p0, sim_turb_df.u_p0, atol=0.01)


def test_gen_turb_error_con_data():
    """verify the warnings are thrown"""
    # given
    con_spat_df = pd.DataFrame([[0, 0, 0, 0, 75]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[9], [11]], index=[0, 1], columns=['u_p0'])
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 75, 'T': 2,
              'nt': 2, 'con_data': con_data}
    # when and then
    with pytest.raises(ValueError):
        gen_turb(spat_df, **kwargs)


def test_gen_turb_bad_interp():
    """verify the errors are thrown for bad interp_data options"""
    # given
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 75, 'T': 2,
              'nt': 2}
    con_spat_df = pd.DataFrame([[0, 0, 0, 0, 75]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[0], [1]], index=[9, 11], columns=['u_p0'])
    con_tc = TimeConstraint().from_con_data(con_spat_df=con_spat_df, con_turb_df=con_turb_df)
    # when and then
    with pytest.raises(ValueError):  # no con_tc given
        gen_turb(spat_df, interp_data='all', **kwargs)
    with pytest.raises(ValueError):  # bad string
        gen_turb(spat_df, interp_data='dog', con_tc=con_tc, **kwargs)
    with pytest.raises(ValueError):  # bad string in list
        gen_turb(spat_df, interp_data=['dog'], con_tc=con_tc, **kwargs)


def test_gen_turb_bad_con_tc():
    """verify the errors are thrown when con_tc dt doesn't match sim dt"""
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 75, 'T': 1,
              'nt': 2}
    con_spat_df = pd.DataFrame([[0, 0, 0, 0, 75]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[9], [11]], index=[0, 5], columns=['u_p0'])
    con_tc = TimeConstraint().from_con_data(con_spat_df=con_spat_df, con_turb_df=con_turb_df)
    # when and then
    with pytest.raises(ValueError):  # no con_tc given
        gen_turb(spat_df, con_tc=con_tc, **kwargs)


def test_gen_turb_wsp_func():
    """verify that you recover the custom mean wind speed"""
    # given
    spat_df = pd.DataFrame([[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [50, 50, 50, 90]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0', 'u_p1'])
    wsp_func = lambda spat_df, **kwargs: 4 * np.isclose(spat_df.loc['k'], 0)  # constant wind speed
    kwargs = {'u_ref': 10}
    u_theory = np.array([4, 0, 0, 4])
    # when
    wsp_vals = gen_turb(spat_df, wsp_func=wsp_func, **kwargs).mean().values
    # then
    np.testing.assert_almost_equal(u_theory, wsp_vals)


def test_gen_turb_sig_func():
    """verify that you recover the custom turb. standard deviation"""
    # given
    spat_df = pd.DataFrame([[0, 1, 2], [0, 0, 0], [0, 0, 0], [50, 50, 50]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0'])
    sig_theory = np.array([1, 1, 1])
    sig_func = lambda spat_df, **kwargs: np.ones_like(spat_df.loc['y'])  # constant std dev
    kwargs = {'u_ref': 10, 'T': 1, 'nt': 4}
    # when
    turb = gen_turb(spat_df, sig_func=sig_func, **kwargs)
    sig_vals = np.std(turb, axis=0, ddof=1)
    # then
    np.testing.assert_almost_equal(sig_theory, sig_vals)


def test_gen_turb_spec_func():
    """verify custom power spectrum recovered. 4 time steps, spec is powers of integers
    so mags should proceed linearly in steps of 0.5 (2-sidedness)."""
    # given
    spat_df = pd.DataFrame([[0, 1, 2], [0, 0, 0], [0, 0, 0], [50, 50, 50]],
                           index=_spat_rownames, columns=['u_p0', 'v_p0', 'w_p0'])
    def spec_func(f, spat_df, **kwargs):
        """square of arange should make magnitudes proceed linearly"""
        return np.tile(np.arange(f.size)**2, (spat_df.shape[1], 1)).T  # tile to match n_inp
    normmag_theory = np.tile(np.linspace(0, 1, 3), (spat_df.shape[1], 1)).T
    kwargs = {'u_ref': 10, 'T': 1, 'nt': 4}
    # when
    turb = gen_turb(spat_df, spec_func=spec_func, **kwargs)
    mags = np.abs(np.fft.rfft(turb.values, axis=0)) / 2  # n_t = 2
    mags[0, :] = 0
    normmag_vals = mags / np.max(mags, axis=0)
    # then
    np.testing.assert_allclose(normmag_theory, normmag_vals)


def test_gen_turb_verbose():
    """make sure the verbose option doesn't break anything"""
    # given
    spat_df = pd.DataFrame(np.ones((4, 1)), index=_spat_rownames)
    kwargs = {'u_ref': 10, 'T': 1, 'nt': 2, 'verbose': True}
    # when
    gen_turb(spat_df, **kwargs)
    # then
    pass


def test_gen_turb_sims_collocated():
    """Should return None if all sim pts collocated with constraints"""
    # given -- constraining points
    con_spat_df = pd.DataFrame([[0, 0, 0, 70]], columns=_spat_rownames).T
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70, 'T': 300,
              'nt': 600, 'seed': 1337}
    coh_model = 'iec'
    con_turb_df = gen_turb(con_spat_df, coh_model=coh_model, **kwargs)
    con_tc = TimeConstraint().from_con_data(con_spat_df=con_spat_df.T, con_turb_df=con_turb_df)
    # given -- simulated, constrainted turbulence
    spat_df = con_spat_df  # simulate same points as constraint
    # when
    sim_turb_df = gen_turb(spat_df, con_tc=con_tc, coh_model=coh_model, **kwargs)
    # then (std dev, mean, and regen'd time series should be close; right colnames)
    assert sim_turb_df is None


def test_gen_turb_warning_dt():
    """Should warn if dt given in kwarg"""
    # given
    y, z = 0, [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 75, 'T': 2,
              'dt': 1}
    # when and then
    with pytest.warns(DeprecationWarning):
        gen_turb(spat_df, **kwargs)


def test_gen_turb_same_coh_file(tmp_path):
    """Verify that gen_turb produces the same result with and without the coh_file option"""
    # given
    y, z = [-5, 5], [70, 80]
    spat_df = gen_spat_grid(y, z, comps=[0])
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70, 'T': 6, 'nt': 6,
              'seed': 1337, 'nf_chunk': 2}
    coh_file = tmp_path / 'test.h5'
    # when
    for nf_chunk in [1, 2, 5, 6]:
        t, freq = get_freq(**kwargs)
        turb_df_orig = gen_turb(spat_df, **kwargs)
        generate_coherence_file(spat_df, coh_file, freq=freq, **kwargs)
        turb_df_cohfile = gen_turb(spat_df, coh_file=coh_file, **kwargs)
        # then
        pd.testing.assert_frame_equal(turb_df_orig, turb_df_cohfile)


if __name__ == '__main__':
    test_iec_turb_mn_std_dev()
    test_gen_turb_con()
    test_gen_turb_error_con_data()
    test_gen_turb_bad_interp()
    test_gen_turb_bad_con_tc()
    test_gen_turb_wsp_func()
    test_gen_turb_sig_func()
    test_gen_turb_spec_func()
    test_gen_turb_sims_collocated()
    test_gen_turb_warning_dt()
    # test_gen_turb_same_coh_file()  # uses pytest tmpdir fixture
