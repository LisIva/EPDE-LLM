from scipy.io import loadmat
import time
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import epde.interface.interface as epde
import os
from pathlib import Path
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker
from epde_eq_parse.eq_parser import clean_parsed_out


if __name__ == '__main__':
    dir_name = 'burg_sindy'
    path_full = os.path.join(Path().absolute().parent, "data_burg", "burgers.mat")
    burg = loadmat(path_full)
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    u = np.real(burg['usol'])
    u = np.transpose(u)

    boundary = 10
    dimensionality = u.ndim-1
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 100

    i = 0
    clean_parsed_out(dir_name)
    run_eq_info = []

    epde_search_obj = epde.EpdeSearch(use_solver=False, boundary=boundary,
                                      coordinate_tensors=grids,
                                      prune_domain=False)

    epde_search_obj.set_moeadd_params(population_size=8, training_epochs=7)
    alg_runtime = 0
    while i < max_iter_number:

        start = time.time()
        try:
            epde_search_obj.fit(data=u, max_deriv_order=(1, 2),
                                equation_terms_max_number=3, equation_factors_max_number=2,
                                eq_sparsity_interval=(1e-08, 1e-1), additional_tokens=[])
        except IndexError:
            continue

        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=4)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=4)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best())

        time1 = end-start
        alg_runtime += time1
        print('Iter time is:', time1)
        print('Number of found eqs:', len(iter_info))
        print(f'Iteration processed: {i+1}/{max_iter_number}')
        i += 1

    print('Overall alg time is:', alg_runtime)
    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv()
    print()
