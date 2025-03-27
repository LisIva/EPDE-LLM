import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
from scipy.io import loadmat
import traceback
import logging
import os
from pathlib import Path


if __name__ == '__main__':
    path_full = os.path.join(Path().absolute().parent, "data_kdv", "kdv.mat")
    kdV = loadmat(path_full)
    t = np.ravel(kdV['t'])
    x = np.ravel(kdV['x'])
    u = np.real(kdV['usol'])
    u = np.transpose(u)

    boundary = 0
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 50

    i = 0
    while i < max_iter_number:
        epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                              dimensionality=dimensionality, coordinate_tensors=grids)

        epde_search_obj.set_moeadd_params(population_size=8, training_epochs=90)
        start = time.time()

        epde_search_obj.fit(data=u, max_deriv_order=(1, 3),
                            equation_terms_max_number=4, equation_factors_max_number=2,
                            eq_sparsity_interval=(1e-08, 1e-06))
        epde_search_obj.equations(only_print=True, only_str=False, num=4)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=4)

        end = time.time()
        time1 = end-start
        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}')
        i += 1

