import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import traceback
import logging
import os
from pathlib import Path
from sympy import Mul, Symbol


if __name__ == '__main__':
    path_full = os.path.join(Path().absolute().parent, "data_burg", "burgers.mat")
    burg = loadmat(path_full)
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    u = np.real(burg['usol'])
    u = np.transpose(u)

    boundary = 10
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 1
    i = 0
    alg_time_start = time.time()
    while i < max_iter_number:
        epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                              dimensionality=dimensionality, coordinate_tensors=grids)

        epde_search_obj.set_moeadd_params(population_size=8, training_epochs=7)
        start = time.time()
        epde_search_obj.fit(data=u, max_deriv_order=(1, 2),
                            equation_terms_max_number=3, equation_factors_max_number=2,
                            eq_sparsity_interval=(1e-08, 1e-1))
        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=4)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=4)
        time1 = end-start

        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}')
        i += 1

    alg_time_end = time.time()
    print(f"Time for all runs, min: {(alg_time_end - alg_time_start) / 60: .2f}")
    print(f"Time for all runs, hours: {(alg_time_end - alg_time_start) / 3600: .2f}")
