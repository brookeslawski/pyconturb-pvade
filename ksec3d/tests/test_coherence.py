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

from ksec3d.core.coherence import get_coherence, get_iec_coherence
from ksec3d.core.helpers import gen_spat_grid, spat_to_pair_df
from ksec3d.core.simulation import gen_unc_turb


def test_main_default():
    """Check the default value for get_coherence
    """
    # given
    pair_df = pd.DataFrame([['vxt', 0, 0, 0, 'vxt', 0, 0, 1]],
                           columns=['k1', 'x1', 'y1', 'z1',
                                    'k2', 'x2', 'y2', 'z2'])
    freq = 1
    kwargs = {'v_hub': 1, 'l_c': 1}
    coh_theory = 5.637379774e-6

    # when
    coh = get_coherence(pair_df, freq,
                        **kwargs)

    # then
    assert np.isclose(coh, coh_theory)


def test_main_badcohmodel():
    """Should raise an error if a wrong coherence model is passed in
    """
    # given
    spat_df = pd.DataFrame([['vxt', 0, 0, 0, 'vxt', 0, 0, 1]],
                           columns=['k1', 'x1', 'y1', 'z1',
                                    'k2', 'x2', 'y2', 'z2'])
    freq = 1

    # when & then
    with pytest.raises(ValueError):
        get_coherence(spat_df, freq,
                      coh_model='garbage')


def test_iec_badedition():
    """IEC coherence should raise an error if any edn other than 3 is given
    """
    # given
    spat_df = pd.DataFrame([['vxt', 0, 0, 0, 'vxt', 0, 0, 1]],
                           columns=['k1', 'x1', 'y1', 'z1',
                                    'k2', 'x2', 'y2', 'z2'])
    freq = 1
    kwargs = {'ed': 4, 'v_hub': 12, 'l_c': 340.2}

    # when & then
    with pytest.raises(ValueError):
        get_iec_coherence(spat_df, freq,
                          **kwargs)


def test_iec_missingkwargs():
    """IEC coherence should raise an error if missing parameter(s)
    """
    # given
    spat_df = pd.DataFrame([['vxt', 0, 0, 0, 'vxt', 0, 0, 1]],
                           columns=['k1', 'x1', 'y1', 'z1',
                                    'k2', 'x2', 'y2', 'z2'])
    freq = 1
    kwargs = {'ed': 3, 'v_hub': 12}

    # when & then
    with pytest.raises(ValueError):
        get_iec_coherence(spat_df, freq,
                          **kwargs)


def test_iec_value():
    """Verify that the value of IEC coherence matches theory
    """
    # given
    spat_df = pd.DataFrame([['vxt', 0, 0, 0, 'vxt', 0, 0, 1]],
                           columns=['k1', 'x1', 'y1', 'z1',
                                    'k2', 'x2', 'y2', 'z2'])
    freq = 0.5
    kwargs = {'ed': 3, 'v_hub': 2, 'l_c': 3}
    coh_theory = 0.0479231144

    # when
    coh = get_iec_coherence(spat_df, freq,
                            **kwargs)

    # then
    assert np.isclose(coh, coh_theory)

    # given
    spat_df = pd.DataFrame([['vxt', 0, 0, 0, 'vyt', 0, 0, 1]],
                           columns=['k1', 'x1', 'y1', 'z1',
                                    'k2', 'x2', 'y2', 'z2'])
    freq = 1
    kwargs = {'ed': 3, 'v_hub': 2, 'l_c': 3}
    coh_theory = 0

    # when
    coh = get_iec_coherence(spat_df, freq,
                            **kwargs)

    # then
    assert np.isclose(coh, coh_theory)


@pytest.mark.long  # mark this as a slow test
def test_verify_iec_sim_coherence():
    """check that the simulated box has the right coherence
    """
    # given
    y, z = [0], [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2, 'z_hub': 70,
              'T': 300, 'dt': 100}
    coh_model, spc_model = 'iec', 'kaimal'
    n_real = 1000  # number of realizations in ensemble
    coh_thresh = 0.12  # coherence threshold

    # get theoretical coherence
    pair_df = spat_to_pair_df(spat_df)
    coh_theo = get_coherence(pair_df, 1 / kwargs['T'],
                             **kwargs).values.flatten()

    # when
    turb_ens = np.empty((int(np.ceil(kwargs['T']/kwargs['dt'])),
                         3 * len(y) * len(z), n_real))
    for i_real in range(n_real):
        turb_ens[:, :, i_real] = gen_unc_turb(spat_df, coh_model=coh_model,
                                              spc_model=spc_model,
                                              **kwargs)
    turb_fft = np.fft.rfft(turb_ens, axis=0)
    ii_jj = [(i, j) for (i, j) in itertools.combinations(spat_df.index, 2)]
    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]
    x_ii, x_jj = turb_fft[1, ii, :], turb_fft[1, jj, :]
    coh = np.mean((x_ii * np.conj(x_jj)) /
                  (np.sqrt(x_ii * np.conj(x_ii)) *
                   np.sqrt(x_jj * np.conj(x_jj))),
                  axis=-1)
    max_coh_diff = np.abs(coh - coh_theo).max()

    # then
    assert max_coh_diff < coh_thresh
