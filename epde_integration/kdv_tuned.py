import os
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker
import numpy as np
import time
from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator
from epde.interface.interface import EpdeSearch
import scipy.io as scio


def noise_data(data, noise_level):
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def kdv_data(filename="KdV_sln_100.csv", shape=100):
    dir_kdv = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "data_kdv")
    data = np.loadtxt(os.path.join(dir_kdv, filename), delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, data


def kdv_sindy_data(filename="kdv.mat"):
    dir_kdv = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "data_kdv")
    data = scio.loadmat(os.path.join(dir_kdv, filename))
    t = np.ravel(data['t'])
    x = np.ravel(data['x'])
    u = np.real(data['usol'])
    u = np.transpose(u)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, u


def kdv_inhomo_discovery(noise_level):
    grid, data = kdv_data()
    # noised_data = data
    noised_data = noise_data(data, noise_level)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=10,
                                      coordinate_tensors=grid, device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                        preprocessor_kwargs={}) #'epochs_max' : 1e3

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=30)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=False)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-5, 1e-2)

    i, max_iter_number = 0, 100
    dir_name = 'kdv'
    run_eq_info = []
    while i < max_iter_number:
        start = time.time()

        epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(1, 3), derivs=None,
                            equation_terms_max_number=5, data_fun_pow=1,
                            additional_tokens=[custom_trig_tokens],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn
        end = time.time()

        epde_search_obj.equations(only_print=True, num=1)
        res = epde_search_obj.equations(only_print=False, num=1)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best())
        i += 1
        print(f"Iter #{i}/{max_iter_number} completed")
        print(f"Time spent: {(end - start) / 60} min")

    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv()


def kdv_sindy_discovery(noise_level):
    dir_name = 'kdv_sindy'
    grid, data = kdv_sindy_data()
    noised_data = noise_data(data, noise_level)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=10,
                                      coordinate_tensors=grid, device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
    bounds = (1e-5, 1e-2)
    i, max_iter_number = 0, 100

    run_eq_info = []
    while i < max_iter_number:
        start = time.time()
        epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(1, 3), derivs=None,
                            equation_terms_max_number=5, data_fun_pow=1,
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn
        end = time.time()

        epde_search_obj.equations(only_print=True, num=3)
        res = epde_search_obj.equations(only_print=False, num=3)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best())
        i += 1
        print(f"Iter #{i}/{max_iter_number} completed")
        print(f"Time spent: {(end - start) / 60} min")
    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv()
    return epde_search_obj


if __name__ == "__main__":
    kdv_sindy_discovery(noise_level=0)
    kdv_inhomo_discovery(noise_level=0)
