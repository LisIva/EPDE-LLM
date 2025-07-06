import os
import time
import numpy as np
from epde.interface.interface import EpdeSearch
from scipy.io import loadmat
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker


def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def burgers_data():
    dir_burg = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "data_burg")
    burg = loadmat(os.path.join(dir_burg, "burgers.mat"))
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    data = np.real(burg['usol'])
    data = np.transpose(data)
    grids = np.meshgrid(t, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    return grids, data


def burgers_discovery(noise_level):
    dir_name = 'burg_sindy'
    grid, data = burgers_data()
    noised_data = noise_data(data, noise_level)

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                  use_pic=True, boundary=20,
                                  coordinate_tensors=grid, device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=15)
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
    bounds = (1e-5, 1e-0)

    i, max_iter_number = 0, 100
    run_eq_info = []
    while i < max_iter_number:
        start = time.time()
        epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                            equation_terms_max_number=5, data_fun_pow=3,
                            additional_tokens=[],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=bounds, fourier_layers=False) #
        end = time.time()

        epde_search_obj.equations(only_print=True, num=3)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=3)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best())
        i += 1
        print(f"Iter #{i}/{max_iter_number} completed")
        print(f"Time spent: {(end-start)/60} min")
    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv()


if __name__ == "__main__":
    burgers_discovery(noise_level=0)
#     -1.002749341339657 * du/dx0{power: 1.0} * u{power: 2.0} + 0.0 * du/dx0{power: 1.0} + 0.0 * d^2u/dx0^2{power: 1.0} * d^3u/dx1^3{power: 1.0} + 0.10098401653138121 * u{power: 2.0} * d^2u/dx1^2{power: 1.0} + 1.1827119883039972e-06 = du/dx1{power: 1.0} * u{power: 3.0}
# -0.999411682452236 * du/dx1{power: 1.0} * u{power: 1.0} + 0.00016513501957518897 * d^3u/dx1^3{power: 1.0} + 0.10118281063214939 * d^2u/dx1^2{power: 1.0} + 0.0 * d^2u/dx0^2{power: 1.0} * d^3u/dx1^3{power: 1.0} + 6.686071463643607e-06 = du/dx0{power: 1.0}
# {'terms_number': {'optimizable': False, 'value': 5}, 'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 'probas': [0.65, 0.35]}}, ('sparsity', 'u'): {'optimizable': True, 'value': 0.0019026364197183264}} , with objective function values of [0.02226763 0.00108123]
