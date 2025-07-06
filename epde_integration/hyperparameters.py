from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens
import numpy as np


custom_trigonometric_eval_fun = {
    'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                        eval_fun_params_labels=['power'])
trig_params_ranges = {'power': (1, 1)}
trig_params_equal_ranges = {}

custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                  token_labels=['cos(t)sin(x)'],
                                  evaluator=custom_trig_evaluator,
                                  params_ranges=trig_params_ranges,
                                  params_equality_ranges=trig_params_equal_ranges,
                                  meaningful=True, unique_token_type=False)

epde_params = {
    'burg': {'boundary': 10,
             'population_size': 5,
             'training_epochs': 5,
             'max_deriv_order': (1, 1),
             'equation_terms_max_number': 3,
             'equation_factors_max_number': 2,
             'eq_sparsity_interval': (1e-08, 1e-4),
             'num': 4,
             'additional_tokens': None,
             "use_pic": False,
             "data_fun_pow": 1,
             "fourier_layers": True},

    'burg_sindy': {'boundary': 20,
                   'population_size': 8,
                   'training_epochs': 15,
                   'max_deriv_order': (2, 3),
                   'equation_terms_max_number': 5,
                   'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
                   'eq_sparsity_interval': (1e-5, 1e-0),
                   'num': 3,
                   'additional_tokens': None,
                   "use_pic": True,
                   "data_fun_pow": 3,
                   "fourier_layers": False},

    'kdv': {'boundary': 10,
            'population_size': 8,
            'training_epochs': 30,
            'max_deriv_order': (1, 3),
            'equation_terms_max_number': 5,
            'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
            'eq_sparsity_interval': (1e-5, 1e-2),
            'num': 2,
            'additional_tokens': custom_trig_tokens,
             "use_pic": True,
             "data_fun_pow": 1,
             "fourier_layers": False},

    'kdv_sindy': {'boundary': 10,
                  'population_size': 8,
                  'training_epochs': 5,
                  'max_deriv_order': (1, 3),
                  'equation_terms_max_number': 5,
                  'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
                  'eq_sparsity_interval': (1e-5, 1e-2),
                  'num': 3,
                  'additional_tokens': None,
                  "use_pic": True,
                  "data_fun_pow": 1,
                  "fourier_layers": False},

    'wave': {'boundary': 10,
             'population_size': 5,
             'training_epochs': 5,
             'max_deriv_order': (2, 2),
             'equation_terms_max_number': 3,
             'equation_factors_max_number': 1,
             'eq_sparsity_interval': (1e-08, 5),
             'num': 2,
             'additional_tokens': None,
             "use_pic": False,
             "data_fun_pow": 1,
             "fourier_layers": True},
}