import epde.interface.interface as epde
from epde.interface.equation_translator import translate_equation
from epde.interface.prepared_tokens import ControlVarTokens
import numpy as np
import torch
import traceback
import logging
import os
from pathlib import Path
import pandas as pd
import time


def translate_dummy_eqs(t: np.ndarray, x: np.ndarray, u: np.ndarray, eq_u: str, diff_method='FD', bnd=10, device='cpu'):
    dimensionality = u.ndim - 1

    epde_search_obj = epde.EpdeSearch(use_solver=False, dimensionality=dimensionality, boundary=bnd,
                                      coordinate_tensors=[t, x], verbose_params={'show_iter_idx': True},
                                      device=device)

    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max': 50000, 'device': device})
    elif diff_method == 'poly':
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                                                              'polynomial_window': 3, 'poly_order': 4})
    elif diff_method == 'FD':
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={})
    else:
        raise ValueError('Incorrect preprocessing tool selected.')
    epde_search_obj.set_moeadd_params(population_size=5, training_epochs=5)
    epde_search_obj.create_pool(data=u, max_deriv_order=(2, 2), additional_tokens=[],
                                data_fun_pow=1, deriv_fun_pow=1,
                                derivs=None)
    return translate_equation(eq_u, pool=epde_search_obj.pool, all_vars=['u',])


if __name__ == '__main__':
    path_full = os.path.join(Path().absolute(), "data_wave", "wave_sln_100.csv")
    df = pd.read_csv(path_full, header=None)
    u = df.values
    u = np.transpose(u)
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)

    boundary = 10
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')
    # eq_u = '20. * u{power: 1} + -1. * d^2u/dx0^2{power: 1} + 0 = d^2u/dx1^2{power: 1}'
    eq_u = '0.017083710392065486 * u{power: 1.0} + 0.0004237250964155818 * du/dx1{power: 1.0} + -1.0586354035340615 = d^2u/dx0^2{power: 1.0}'
    # -1.0586354035340615 * du/dx1{power: 1.0} + 0.0004237250964155818 * u{power: 1.0} + 0.017083710392065486 = d^2u/dx0^2{power: 1.0}
    # 0.017083710392065486 * u{power: 1.0} + 0.0004237250964155818 * du/dx1{power: 1.0} + -1.0586354035340615 = d^2u/dx0^2{power: 1.0}
    # {'sparsity': {'optimizable': True, 'value': 1.0}, 'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': 1}}
    # start = time.time()
    test = translate_dummy_eqs(grids[0], grids[1], u, eq_u, diff_method='FD', bnd=boundary)
    end = time.time()
    # time1 = end - start
    # print('Overall time is:', time1)
    print()
