import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde
from scipy.io import loadmat
import traceback
import logging
import os
from pathlib import Path
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker
from epde_eq_parse.eq_parser import clean_parsed_out
import pickle

# -0.17705425481602444 * d^3u/dx1^3{power: 1.0} + -0.16377446251841735 * du/dx0{power: 1.0} + 0.0 * du/dx0{power: 1.0} * du/dx1{power: 1.0} + -2.057181974238461e-06 = du/dx1{power: 1.0} * u{power: 1.0}
# -6.099944218365907 * du/dx1 * u + 0.0 * d^3u/dx1^3 * du/dx0 + -1.0793047739940058 * d^3u/dx1^3 + -1.1845818829292476e-05 = du/dx0
if __name__ == '__main__':
    dir_name = 'kdv_sindy'
    path_full = os.path.join(Path().absolute().parent, "data_kdv", "kdv.mat")
    kdV = loadmat(path_full)
    t = np.ravel(kdV['t'])
    x = np.ravel(kdV['x'])
    u = np.real(kdV['usol'])
    u = np.transpose(u)

    boundary = 0
    dimensionality = u.ndim-1
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 2

    i = 0
    clean_parsed_out(dir_name)
    run_eq_info = []

    epde_search_obj = epde.EpdeSearch(use_solver=False, boundary=boundary,
                                      dimensionality=dimensionality, coordinate_tensors=grids)

    epde_search_obj.set_moeadd_params(population_size=8, training_epochs=90)

    while i < max_iter_number:

        start = time.time()
        try:
            epde_search_obj.fit(data=u, max_deriv_order=(1, 3),
                                equation_terms_max_number=4, equation_factors_max_number=2,
                                eq_sparsity_interval=(1e-08, 1e-06))
        except IndexError:
            continue

        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=4)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=4)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)

        with open('res.pickle', 'wb') as file:
            pickle.dump(res, file)

        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best())

        time1 = end-start
        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}')
        i += 1

    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv()
    print()
