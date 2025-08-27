import os
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker
import numpy as np
import time
from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator
from epde.interface.interface import EpdeSearch
import scipy.io as scio


def noise_data(data, noise_level):
    return noise_level * np.std(data) * np.random.normal(size=data.shape) + data

def kdv_sindy_data(filename="kdv.mat"):
    dir_kdv = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "data_kdv")
    data = scio.loadmat(os.path.join(dir_kdv, filename))
    t = np.ravel(data['t'])
    x = np.ravel(data['x'])
    u = np.real(data['usol'])
    u = np.transpose(u)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, u

def kdv_sindy_discovery(noise_level, epochs):
    dir_name = 'kdv_sindy'
    experiment_info = "noise_" + str(noise_level) + "_epochs_" + str(epochs)
    grid, data = kdv_sindy_data()

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
    bounds = (1e-5, 1e0)
    i, max_iter_number = 0, 30

    run_eq_info = []
    while i < max_iter_number:
        noised_data = noise_data(data, noise_level)
        epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                     boundary=10,
                                     coordinate_tensors=grid, device='cuda')

        if noise_level == 0:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                             preprocessor_kwargs={})
        else:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                             preprocessor_kwargs={"use_smoothing": True})

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize,
                                          training_epochs=epochs)

        start = time.time()
        epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                            equation_terms_max_number=5, data_fun_pow=3,
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn
        end = time.time()

        epde_search_obj.equations(only_print=True, num=1)
        res = epde_search_obj.equations(only_print=False, num=1)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best("shd"))
        i += 1
        print(f"Iter #{i}/{max_iter_number} completed")
        print(f"Time spent: {(end - start) / 60} min")
    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv(package="epde_experiments", experiment_info=experiment_info)
    return epde_search_obj


if __name__ == "__main__":
    ''' Parameters of the experiment '''
    epochs = 25
    noise_level = 0
    ''''''
    kdv_sindy_discovery(noise_level=noise_level, epochs=epochs)
