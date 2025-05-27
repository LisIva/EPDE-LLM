class WaveSch(object):
    """
    Static class to store wave equation schema and parameters.
    """
    left_side = 'd^2u/dx0^2'
    schema = frozenset({'d^2u/dx0^2', 'd^2u/dx1^2', 'C'})
    correct_params1 = {'d^2u/dx0^2': -1.0, 'd^2u/dx1^2': 0.04, 'C': 0.0}
    correct_params2 = {'d^2u/dx0^2': -25.0, 'd^2u/dx1^2': 1.0, 'C': 0.0}
    correct_params3 = {'d^2u/dx0^2': 1.0, 'd^2u/dx1^2': -0.04, 'C': 0.0}
    correct_params4 = {'d^2u/dx0^2': 25.0, 'd^2u/dx1^2': -1.0, 'C': 0.0}
    params = [correct_params1, correct_params2, correct_params3, correct_params4]


class BurgSch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 1.0, 'C': 0.0}
    correct_params2 = {'du/dx0': -1.0, 'u * du/dx1': -1.0, 'C': 0.0}
    params = [correct_params1, correct_params2]


class BurgSindySch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^2u/dx1^2', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 1.0, 'd^2u/dx1^2': -0.1, 'C': 0.0}
    correct_params2 = {'du/dx0': 10.0, 'u * du/dx1': 10.0, 'd^2u/dx1^2': -1.0, 'C': 0.0}
    correct_params3 = {'du/dx0': -1.0, 'u * du/dx1': -1.0, 'd^2u/dx1^2': 0.1, 'C': 0.0}
    correct_params4 = {'du/dx0': -10.0, 'u * du/dx1': -10.0, 'd^2u/dx1^2': 1.0, 'C': 0.0}
    params = [correct_params1, correct_params2, correct_params3, correct_params4]


class KdvSindySch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^3u/dx1^3', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 6.0, 'd^3u/dx1^3': 1.0, 'C': 0.0}
    correct_params2 = {'du/dx0': 1/6, 'u * du/dx1': 1.0, 'd^3u/dx1^3': 1/6, 'C': 0.0}
    correct_params3 = {'du/dx0': -1.0, 'u * du/dx1': -6.0, 'd^3u/dx1^3': -1.0, 'C': 0.0}
    correct_params4 = {'du/dx0': -1/6, 'u * du/dx1': -1.0, 'd^3u/dx1^3': -1/6, 'C': 0.0}
    params = [correct_params1, correct_params2, correct_params3, correct_params4]


class KdvSch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^3u/dx1^3', 'cos(t)sin(x)', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 6.0, 'd^3u/dx1^3': 1.0, 'cos(t)sin(x)': -1.0, 'C': 0.0}
    correct_params2 = {'du/dx0': 1/6, 'u * du/dx1': 1.0, 'd^3u/dx1^3': 1/6, 'cos(t)sin(x)': -1/6, 'C': 0.0}
    correct_params3 = {'du/dx0': -1.0, 'u * du/dx1': -6.0, 'd^3u/dx1^3': -1.0, 'cos(t)sin(x)': 1.0, 'C': 0.0}
    correct_params4 = {'du/dx0': -1/6, 'u * du/dx1': -1.0, 'd^3u/dx1^3': -1/6, 'cos(t)sin(x)': 1/6, 'C': 0.0}
    params = [correct_params1, correct_params2, correct_params3, correct_params4]


schemas = {'wave': {'schema': WaveSch.schema,
                    'params': WaveSch.params,
                    'left_side': WaveSch.left_side,},

           'burg_sindy': {'schema': BurgSindySch.schema,
                          'params': BurgSindySch.params,
                          'left_side': BurgSindySch.left_side,},

           'burg': {'schema': BurgSch.schema,
                    'params': BurgSch.params,
                    'left_side': BurgSch.left_side,},

           'kdv': {'schema': KdvSch.schema,
                   'params': KdvSch.params,
                   'left_side': KdvSch.left_side,},

           'kdv_sindy': {'schema': KdvSindySch.schema,
                         'params': KdvSindySch.params,
                         'left_side': KdvSindySch.left_side,},
           }
