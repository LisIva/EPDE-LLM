import sys
sys.path.append("./EPDE/epde")
import time
import numpy as np
import pandas as pd
# import epde.interface.interface as epde
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import traceback
import logging
import os
from pathlib import Path
from epde_integration.hyperparameters import epde_params
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker
from epde_eq_parse.eq_parser import clean_parsed_out
from epde.interface.interface import EpdeSearch

def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * np.std(data) * np.random.normal(size=data.shape) + data

def wave_data():
    base_path = Path().absolute().parent
    path_full = os.path.join(base_path, "data_wave", "wave_sln_100.csv")
    df = pd.read_csv(path_full, header=None)
    data = df.values
    data = np.transpose(data)
    t = np.linspace(0, 1, 101)
    x = np.linspace(0, 1, 101)
    grids = np.meshgrid(t, x, indexing='ij')
    return data, grids

def wave_discovery(noise_level, epochs):
    data, grids = wave_data()

    i = 0
    max_iter_number = 30
    clean_parsed_out('wave')
    run_eq_info = []

    boundary = 20
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    while i < max_iter_number:
        noised_data = noise_data(data, noise_level)
        epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=boundary,
                                     coordinate_tensors=grids,
                                     prune_domain=False,
                                     device='cuda')
        if noise_level == 0:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                             preprocessor_kwargs={})
        else:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                             preprocessor_kwargs={"use_smoothing": True})  # "use_smoothing": True

        epde_search_obj.set_moeadd_params(population_size=8, training_epochs=epochs)

        start = time.time()

        try:
            epde_search_obj.fit(data=noised_data, max_deriv_order=(2, 3),
                                equation_terms_max_number=5, data_fun_pow=3,
                                equation_factors_max_number=factors_max_number,
                                eq_sparsity_interval=(1e-5, 1e0), additional_tokens=[])
        except IndexError:
            continue
        end = time.time()
        epde_search_obj.equations(only_print=True, only_str=False, num=1)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=1)
        iter_info = evaluate_fronts(res, 'wave', end - start, i)
        run_eq_info += iter_info

        time1 = end - start
        print('Overall time is:', time1)
        print(f'Iteration processed: {i + 1}/{max_iter_number}\n')
        i += 1

    eq_r = EqReranker(run_eq_info, 'wave')
    best_info = eq_r.select_best('shd')
    experiment_info = "noise_" + str(noise_level) + "_epochs_" + str(epochs)
    eq_r.to_csv(package="epde_experiments", experiment_info=experiment_info)
    print()

if __name__ == '__main__':
    ''' Parameters of the experiment '''
    epochs = 5
    noise_level = 0
    ''''''
    wave_discovery(noise_level=noise_level, epochs=epochs)




