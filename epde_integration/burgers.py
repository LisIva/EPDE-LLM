import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import traceback
import logging
import os
from pathlib import Path

# А с equation_factors_max_number получается мы считаем что коэффициент перед слагаемым в него не входит
# pareto_nums
if __name__ == '__main__':

    path = "data_burg"
    path_full = os.path.join(Path().absolute().parent, path, "burgers_sln_100.csv")
    df = pd.read_csv(path_full, header=None)

    u = df.values
    u = np.transpose(u)
    x = np.linspace(-1000, 0, 101)
    t = np.linspace(0, 1, 101)

    boundary = 10
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    max_iter_number = 1

    i = 0
    while i < max_iter_number:
        epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                              dimensionality=dimensionality, coordinate_tensors=grids)

        epde_search_obj.set_moeadd_params(population_size=5, training_epochs=5)
        start = time.time()
        epde_search_obj.fit(data=u, max_deriv_order=(1, 1),
                            equation_terms_max_number=3, equation_factors_max_number=2,
                            eq_sparsity_interval=(1e-08, 1e-4), additional_tokens=[])
        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=4)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=4)
        time1 = end-start

        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}\n')
        i += 1
