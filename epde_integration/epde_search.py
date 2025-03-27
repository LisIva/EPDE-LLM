import epde.interface.interface as epde
from epde.interface.equation_translator import translate_equation


def get_epde_search_obj(grids, dir_name, diff_method='FD'):
    dimensionality = grids[0].ndim-1

    epde_search_obj = epde.EpdeSearch(use_solver=False, dimensionality=dimensionality,
                                      boundary=epde_params[dir_name]['boundary'],
                                      coordinate_tensors=grids, verbose_params={'show_iter_idx': True},
                                      device='cpu')
    epde_search_obj.set_moeadd_params(population_size=epde_params[dir_name]['population_size'],
                                      training_epochs=epde_params[dir_name]['training_epochs'])
    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                         preprocessor_kwargs={'epochs_max': 50000, 'device': 'cpu'})
    elif diff_method == 'poly':
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                                                              'polynomial_window': 3, 'poly_order': 4})
    elif diff_method == 'FD':
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={})
    else:
        raise ValueError('Incorrect preprocessing tool selected.')
    return epde_search_obj