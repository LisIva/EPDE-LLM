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
             'additional_tokens': None},

    'burg_sindy': {'boundary': 10,
                   'population_size': 8,
                   'training_epochs': 7,
                   'max_deriv_order': (1, 2),
                   'equation_terms_max_number': 3,
                   'equation_factors_max_number': 2,
                   'eq_sparsity_interval': (1e-08, 1e-1),
                   'num': 4,
                   'additional_tokens': None},

    'kdv': {'boundary': 0,
            'population_size': 8,
            'training_epochs': 90,
            'max_deriv_order': (1, 3),
            'equation_terms_max_number': 4,
            'equation_factors_max_number': 2,
            'eq_sparsity_interval': (1e-08, 1e-06),
            'num': 2,
            'additional_tokens': custom_trig_tokens},

    'kdv_sindy': {'boundary': 0,
                  'population_size': 8,
                  'training_epochs': 90,
                  'max_deriv_order': (1, 3),
                  'equation_terms_max_number': 4,
                  'equation_factors_max_number': 2,
                  'eq_sparsity_interval': (1e-08, 1e-06),
                  'num': 4,
                  'additional_tokens': None},

    'wave': {'boundary': 10,
             'population_size': 5,
             'training_epochs': 5,
             'max_deriv_order': (2, 2),
             'equation_terms_max_number': 3,
             'equation_factors_max_number': 1,
             'eq_sparsity_interval': (1e-08, 5),
             'num': 2,
             'additional_tokens': None},
}