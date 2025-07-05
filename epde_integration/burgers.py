import time
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import epde.interface.interface as epde
import traceback
import logging
import os
from pathlib import Path
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker
from epde_eq_parse.eq_parser import clean_parsed_out


# А с equation_factors_max_number получается мы считаем что коэффициент перед слагаемым в него не входит
# pareto_nums
if __name__ == '__main__':

    path = "data_burg"
    dir_name = 'burg'
    path_full = os.path.join(Path().absolute().parent, path, "burgers_sln_100.csv")
    df = pd.read_csv(path_full, header=None)

    u = df.values
    u = np.transpose(u)
    x = np.linspace(-1000, 0, 101)
    t = np.linspace(0, 1, 101)

    boundary = 10
    dimensionality = u.ndim-1
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 50

    i = 0
    clean_parsed_out(dir_name)
    run_eq_info = []

    epde_search_obj = epde.EpdeSearch(use_solver=False, boundary=boundary,
                                      coordinate_tensors=grids,
                                      prune_domain=False)

    epde_search_obj.set_moeadd_params(population_size=5, training_epochs=5)

    while i < max_iter_number:
        start = time.time()
        try:
            epde_search_obj.fit(data=u, max_deriv_order=(1, 1),
                                equation_terms_max_number=3, equation_factors_max_number=2,
                                eq_sparsity_interval=(1e-08, 1e-4), additional_tokens=[])
        except IndexError:
            continue

        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=4)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=4)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        if len(iter_info) != 0:
            front_r = FrontReranker(iter_info)
            run_eq_info.append(front_r.select_best())

        time1 = end-start
        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}\n')
        i += 1

    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv()
    print()
