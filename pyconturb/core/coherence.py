# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""
import itertools

import numpy as np


def get_coh_mat(freq, spat_df, coh_model='iec',
                **kwargs):
    """Create coherence matrix for given frequencies and coherence model
    """
    if coh_model == 'iec':  # IEC coherence model
        if 'ed' not in kwargs.keys():  # add IEC ed to kwargs if not passed in
            kwargs['ed'] = 3
        coh_mat = get_iec_coh_mat(freq, spat_df, **kwargs)

    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized.')

    return coh_mat


def get_iec_coh_mat(freq, spat_df,
                    **kwargs):
    """Create IEC 61400-1 Ed. 3 coherence matrix for given frequencies
    """
    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['v_hub', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')

    freq = np.array(freq).reshape(1, -1)
    n_f, n_s = freq.size, spat_df.shape[0]
    ii_jj = [(i, j) for (i, j) in itertools.combinations(spat_df.index, 2)]
    ii = np.array([tup[0] for tup in ii_jj])
    jj = np.array([tup[1] for tup in ii_jj])
    xyz = spat_df[['x', 'y', 'z']].values
    coh_mat = np.repeat(np.eye((n_s)).reshape(n_s, n_s, 1),
                        n_f, axis=2)
    r = np.sqrt((xyz[ii, 1] - xyz[jj, 1])**2 +
                (xyz[ii, 2] - xyz[jj, 2])**2)
    mask = ((spat_df.loc[ii, 'k'].values == 'vxt') &
            (spat_df.loc[jj, 'k'].values == 'vxt'))
    coh_values = np.exp(-12 * np.sqrt((r[mask].reshape(-1, 1) /
                                       kwargs['v_hub'] * freq)**2
                                      + (0.12 * r[mask].reshape(-1, 1) /
                                         kwargs['l_c'])**2))
    coh_mat[ii[mask], jj[mask], :] = coh_values
    coh_mat[jj[mask], ii[mask]] = np.conj(coh_values)
    return coh_mat
