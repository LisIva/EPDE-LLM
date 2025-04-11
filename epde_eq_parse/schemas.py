class WaveSch(object):
    """
    Static class to store wave equation schema and parameters.
    """
    schema = frozenset({'d^2u/dx0^2', 'd^2u/dx1^2', 'C'})
    correct_params1 = {'d^2u/dx0^2': -1.0, 'd^2u/dx1^2': 0.04, 'C': 0.0}
    correct_params2 = {'d^2u/dx0^2': -25.0, 'd^2u/dx1^2': 1.0, 'C': 0.0}
    params = [correct_params1, correct_params2]


class BurgSch(object):
    schema = frozenset({'du/dx0', 'u * du/dx1', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 1.0, 'C': 0.0}
    params = [correct_params1, ]


class BurgSindySch(object):
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^2u/dx1^2', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 1.0, 'd^2u/dx1^2': -0.1, 'C': 0.0}
    correct_params2 = {'du/dx0': 10.0, 'u * du/dx1': 10.0, 'd^2u/dx1^2': -1.0, 'C': 0.0}
    params = [correct_params1, correct_params2]


class KdvSindySch(object):
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^3u/dx1^3', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 6.0, 'd^3u/dx1^3': 1.0, 'C': 0.0}
    correct_params2 = {'du/dx0': 1/6, 'u * du/dx1': 1.0, 'd^3u/dx1^3': 1/6, 'C': 0.0}
    params = [correct_params1, correct_params2]


class KdvSch(object):
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^3u/dx1^3', 'cos(t)sin(x)', 'C'})
    correct_params1 = {'du/dx0': 1.0, 'u * du/dx1': 6.0, 'd^3u/dx1^3': 1.0, 'cos(t)sin(x)': -1.0, 'C': 0.0}
    correct_params2 = {'du/dx0': 1/6, 'u * du/dx1': 1.0, 'd^3u/dx1^3': 1/6, 'cos(t)sin(x)': -1/6, 'C': 0.0}
    params = [correct_params1, correct_params2]


schemas = {'wave': {'schema': WaveSch.schema,
                    'params': WaveSch.params},

           'burg_sindy': {'schema': BurgSindySch.schema,
                          'params': BurgSindySch.params},

           'burg': {'schema': BurgSch.schema,
                    'params': BurgSch.params},

           'kdv': {'schema': KdvSch.schema,
                   'params': KdvSch.params},

           'kdv_sindy': {'schema': KdvSindySch.schema,
                         'params': KdvSindySch.params},
           }
