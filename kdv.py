import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens
import matplotlib.pyplot as plt
import traceback
import logging
import os
from pathlib import Path
import pickle


if __name__ == '__main__':
    base_path = Path().absolute() #.parent
    path_full = os.path.join(base_path, "data_kdv", "KdV_sln_100.csv")
    df = pd.read_csv(path_full, header=None)

    # os.path.join(Path().absolute().parent, "data_kdv", "d_x_100.csv")
    dddx = pd.read_csv(os.path.join(base_path, "data_kdv", "ddd_x_100.csv"), header=None)
    ddx = pd.read_csv(os.path.join(base_path, "data_kdv", "dd_x_100.csv"), header=None)
    dx = pd.read_csv(os.path.join(base_path, "data_kdv", "d_x_100.csv"), header=None)
    dt = pd.read_csv(os.path.join(base_path, "data_kdv", "d_t_100.csv"), header=None)

    u_init = df.values
    u_init = np.transpose(u_init)

    ddd_x = dddx.values
    ddd_x = np.transpose(ddd_x)
    dd_x = ddx.values
    dd_x = np.transpose(dd_x)
    d_x = dx.values
    d_x = np.transpose(d_x)
    d_t = dt.values
    d_t = np.transpose(d_t)

    derivs = np.zeros(shape=(u_init.shape[0],u_init.shape[1],4))
    derivs[:, :, 0] = d_t
    derivs[:, :, 1] = d_x
    derivs[:, :, 2] = dd_x
    derivs[:, :, 3] = ddd_x

    t = np.linspace(0, 1, u_init.shape[0])
    x = np.linspace(0, 1, u_init.shape[1])

    boundary = 0
    dimensionality = u_init.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 5

    start_gl = time.time()

    time_ls = []
    i = 0
    population_error = 0
    while i < max_iter_number:
        epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                               dimensionality=dimensionality, coordinate_tensors=grids)

        custom_trigonometric_eval_fun = {
            'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
        custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                                eval_fun_params_labels=['power'])
        trig_params_ranges = {'power': (1, 1)}
        trig_params_equal_ranges = {}

        custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                          token_labels=['cos(t)sin(x)'],
                                          evaluator=custom_trig_evaluator,
                                          params_ranges=trig_params_ranges,
                                          params_equality_ranges=trig_params_equal_ranges,
                                          meaningful=True, unique_token_type=False)

        epde_search_obj.set_moeadd_params(population_size=8, training_epochs=90)
        start = time.time()
        epde_search_obj.fit(data=u_init, max_deriv_order=(1, 3),
                            equation_terms_max_number=4, equation_factors_max_number=2,
                            eq_sparsity_interval=(1e-08, 1e-06), derivs=[derivs, ],
                            additional_tokens=[custom_trig_tokens, ])
        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=2)
        time1 = end-start
        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}\n')
        time_ls.append(time1)
        i += 1
    end_gl = time.time()
    print(f"Overall time: {(end_gl - start_gl) / 60:.2f}, min.")