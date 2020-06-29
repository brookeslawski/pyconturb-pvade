# -*- coding: utf-8 -*-
"""Test functions in coherence.py

Author
------
Jenni Rinker
rink@dtu.dk
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from pyconturb.simulation import gen_turb
from pyconturb.coherence import get_coh_mat
from pyconturb._utils import gen_spat_grid, _spat_rownames


# ========================== tests for get_coh_mat ==========================

def test_get_coh_mat_default():
    """Check the default value is IEC Ed. 3, standard coh
    """
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 1]], index=_spat_rownames, columns=['u_p0', 'u_p1'])
    freq = 1
    kwargs = {'u_ref': 1, 'l_c': 1}
    coh_theo = np.array([[1, 5.637379774e-6],
                         [5.637379774e-6, 1]])
    # when
    coh_mat = get_coh_mat(freq, spat_df, **kwargs)
    coh_out = coh_mat[0]
    # then
    np.testing.assert_allclose(coh_out, coh_theo)


def test_get_coh_mat_badcohmodel():
    """Should raise an error if a wrong coherence model is passed in
    """
    # given
    spat_df = pd.DataFrame([[0], [0], [0], [0]], index=_spat_rownames, columns=['u_p0'])
    freq = 1
    coh_model = 'garbage'
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(spat_df, freq, coh_model=coh_model)

# ========================== tests for get_iec_coh_mat ==========================

def test_get_iec_coh_mat_badedition():
    """should raise an error if any edn other than 3 or 4 is given
    """
    # given
    spat_df = pd.DataFrame([[0], [0], [0], [0]], index=_spat_rownames, columns=['u_p0'])
    freq, coh_model = 1, 'iec'
    # when
    eds = [3, 4, 5]
    for ed in eds:
        kwargs = {'ed': ed, 'u_ref': 12, 'l_c': 340.2}
        if ed > 4:  # expect error
            with pytest.raises(ValueError):
                get_coh_mat(freq, spat_df, coh_model=coh_model, **kwargs)
        else:
            get_coh_mat(freq, spat_df, coh_model=coh_model, **kwargs)


def test_get_iec_coh_mat_missingkwargs():
    """IEC coherence should raise an error if missing parameter(s)
    """
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 1]], index=_spat_rownames, columns=['u_p0', 'u_p1'])
    freq, kwargs, coh_model = 1, {'ed': 3, 'u_ref': 12}, 'iec'  # missing l_c
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(freq, spat_df, coh_model=coh_model, **kwargs)


def test_get_iec_coh_mat_value_dtype():
    """Verify that the value and datatype of IEC coherence matches theory
    """
    # given (for both sets of functions)
    spat_df = pd.DataFrame([[0, 1, 2, 0, 1, 2],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1]],
                           index=_spat_rownames)
    # -------------------- standard IEC (no 3d), mult freqs --------------------
    coh_model = 'iec'
    dtypes = [np.float64, np.float32]
    for dtype in dtypes:
        # given
        freq, u_ref, l_c = [0.5, 1], 2, 3
        coh_theo = np.tile(np.eye(6), (2, 1, 1))
        for i in range(2):
            coh_theo[i, 0, 3] = np.exp(-12*np.sqrt((freq[i]/u_ref)**2+(0.12/l_c)**2))
            coh_theo[i, 3, 0] = np.exp(-12*np.sqrt((freq[i]/u_ref)**2+(0.12/l_c)**2))
        kwargs = {'ed': 3, 'u_ref': u_ref, 'l_c': l_c, 'coh_model': coh_model}
        # when
        coh_out = get_coh_mat(freq, spat_df, dtype=dtype, **kwargs)
        # then
        np.testing.assert_allclose(coh_out, coh_theo, atol=1e-6)
        assert coh_out.dtype == dtype


@pytest.mark.slow  # mark this as a slow test
@pytest.mark.skipci  # don't run in CI
def test_get_iec_coh_mat_verify_coherence():
    """check that the simulated box has the right coherence (requires many simulations)
    """
    # given
    y, z = [0], [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70,
              'T': 300, 'dt': 100, 'ed': 3}
    coh_model = 'iec'
    n_real = 1000  # number of realizations in ensemble
    coh_thresh = 0.12  # coherence threshold
    # get theoretical coherence
    idcs = np.triu_indices(spat_df.shape[1], k=1)
    coh_theo = get_coh_mat(1 / kwargs['T'], spat_df, coh_model=coh_model,
                           **kwargs)[0][idcs]
    # when
    ii_jj = [(i, j) for (i, j) in
             itertools.combinations(np.arange(spat_df.shape[1]), 2)]  # pairwise indices
    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]
    turb_ens = np.empty((int(np.ceil(kwargs['T']/kwargs['dt'])),
                         3 * len(y) * len(z), n_real))
    for i_real in range(n_real):
        turb_ens[:, :, i_real] = gen_turb(spat_df, coh_model=coh_model, **kwargs)
    turb_fft = np.fft.rfft(turb_ens, axis=0)
    x_ii, x_jj = turb_fft[1, ii, :], turb_fft[1, jj, :]
    coh = np.mean((x_ii * np.conj(x_jj)) /
                  (np.sqrt(x_ii * np.conj(x_ii)) *
                   np.sqrt(x_jj * np.conj(x_jj))),
                  axis=-1)
    max_coh_diff = np.abs(coh - coh_theo).max()
    # then
    assert max_coh_diff < coh_thresh

# ========================== tests for get_iec3d_coh_mat ==========================


def test_get_iec3d_coh_mat_value_dtype():
    """Verify that the value and datatype of 3d coherence matches theory
    """
    spat_df = pd.DataFrame([[0, 1, 2, 0, 1, 2],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1]],
                            index=_spat_rownames)
    
    # -------------------- IEC 3d, singe freq --------------------
    coh_model = 'iec3d'
    dtypes = [np.float64, np.float32]
    for dtype in dtypes:
        # given
        spat_df = pd.DataFrame([[0, 1, 2, 0, 1, 2],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 1]],
                                index=_spat_rownames)
        # spat_df = pd.DataFrame([[comp, comp], [0, 0], [0, 0], [0, 1]],
        #                         index=_spat_rownames, columns=['x_p0', 'y_p1'])
        freqs, u_ref, l_c = 0.5, 2, 3
        kwargs = {'ed': 3, 'u_ref': u_ref, 'l_c': l_c, 'coh_model': coh_model}
        # coh_theo = np.array([[1., coh2], [coh2, 1.]])
        coh_theo = np.eye(6)
        coh_theo[0, 3] = np.exp(-12*np.sqrt((freqs/u_ref)**2+(0.12/l_c)**2))
        coh_theo[3, 0] = coh_theo[0, 3]
        coh_theo[1, 4] = np.exp(-12*np.sqrt((freqs/u_ref)**2+(0.12/l_c*8.1/2.7)**2))
        coh_theo[4, 1] = coh_theo[1, 4]
        coh_theo[2, 5] = np.exp(-12*np.sqrt((freqs/u_ref)**2+(0.12/l_c*8.1/0.66)**2))
        coh_theo[5, 2] = coh_theo[2, 5]
        # when
        coh_mat = get_coh_mat(freqs, spat_df, dtype=dtype, **kwargs)
        coh_out = coh_mat[0]
        # then
        np.testing.assert_allclose(coh_out, coh_theo, atol=1e-6)
        assert coh_out.dtype == dtype


def test_get_iec3d_coh_mat_missingkwargs():
    """IEC coherence should raise an error if missing parameter(s)
    """
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 1]], index=_spat_rownames, columns=['u_p0', 'u_p1'])
    freq, kwargs, coh_model = 1, {'ed': 3, 'u_ref': 12}, 'iec3d'  # missing l_c
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(freq, spat_df, coh_model=coh_model, **kwargs)


if __name__ == '__main__':
    test_get_coh_mat_default()
    test_get_coh_mat_badcohmodel()
    test_get_iec_coh_mat_badedition()
    test_get_iec_coh_mat_missingkwargs()
    test_get_iec_coh_mat_value_dtype()
    test_get_iec3d_coh_mat_value_dtype()    
