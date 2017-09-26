# -*- coding: utf-8 -*-
"""Functions related to the simulation of turbulence
"""

import numpy as np
import pandas as pd

from .coherence import get_coherence
from .spectra import get_spectrum


def gen_turb(spat_df,
             coh_model='iec', spc_model='kaimal', T=600, dt=0.1, **kwargs):
    """Generate turbulence box
    """

    # define time vector
    n_t = np.ceil(T / dt)
    t = np.arange(n_t) * dt

    # create dataframe with magnitudes
    mag_df = get_magnitudes(spat_df, coh_model=coh_model, spc_model=spc_model,
                            T=T, dt=dt, **kwargs)

    # create dataframe with phases
    pha_df = get_phases(spat_df, coh_model=coh_model, spc_model=spc_model,
                        T=T, dt=dt, **kwargs)

    # multiply dataframes together
    turb_fft = mag_df * np.exp(1j * pha_df)

    # inverse fft
    turb_df = pd.DataFrame(np.fft.irfft(turb_fft, axis=1),
                           columns=t)

    return turb_df


def get_magnitudes(spat_df,
                   spc_model='kaimal', T=600, dt=0.1, **kwargs):
    """Create dataframe of magnitudes with desired power spectra
    """

    n_f = np.ceil(T / dt)//2 + 1
    freq = np.arange(n_f) / T
    spc_df = get_spectrum(spat_df, freq, spc_model=spc_model, **kwargs)
    return np.sqrt(spc_df / 2)


def get_phases(spat_df,
               coh_model='iec', T=600, dt=0.1, **kwargs):
    """Create realization of phases with desired coherence
    """

    n_f = np.ceil(T / dt)//2 + 1
    freq = np.arange(n_f) / T
    for i_df, (i, j) in enumerate([(i, j) for i in range(spat_df.shape[0])
                                   for j in range(spat_df.shape[0])]):
        inp_df.iloc[i_df, ['k1', 'x1', 'y1', 'z1']] = spat_df.iloc[i, :]
        inp_df.iloc[i_df, ['k2', 'x2', 'y2', 'z2']] = spat_df.iloc[j, :]
    coh_df = get_coherence(spat_df, freq, coh_model='iec', **kwargs)
