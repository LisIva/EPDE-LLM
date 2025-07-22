import epde.interface.interface as epde
from epde.interface.equation_translator import translate_equation
from epde_integration.hyperparameters import epde_params
from pipeline.epde_translator.sol_track_translator import SolTrackTranslator
import time
from epde_eq_parse.eq_evaluator import evaluate_fronts, FrontReranker
from epde_eq_parse.eq_parser import clean_parsed_out


def get_epde_search_obj(grids, dir_name, device='cpu'):
    epde_search_obj = epde.EpdeSearch(use_solver=False,
                                      use_pic=epde_params[dir_name]['use_pic'],
                                      boundary=epde_params[dir_name]['boundary'],
                                      coordinate_tensors=grids, verbose_params={'show_iter_idx': True},
                                      device=device)

    epde_search_obj.set_moeadd_params(population_size=epde_params[dir_name]['population_size'],
                                      training_epochs=epde_params[dir_name]['training_epochs'])

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    return epde_search_obj


class EpdeSearcher(object):
    # если нужны деривы, то передать вот тут в инит
    def __init__(self, data: list, record_track: dict, pruned_track: dict, dir_name: str,
                 use_init_population=True, max_iter_num=1, device: str = 'cpu'):
        self.__max_iter = max_iter_num
        self.use_init_population = use_init_population
        self.u = data[2]
        self.grids = [data[0], data[1]]
        self._dir_name = dir_name
        self._device = device

        stt = SolTrackTranslator(record_track, pruned_track, dir_name)
        self.__eq_epde_str = stt.translate()
        self.llm_pool = stt.llm_pool
        self.__additional_classes, lambda_strs = self.llm_pool.to_epde_classes()

        self.epde_search_obj = None
        self.population = None

    def __get_max_deriv_order(self):
        max_t = max(epde_params[self._dir_name]['max_deriv_order'][0], self.llm_pool.max_deriv_orders['max_deriv_t'])
        max_x = max(epde_params[self._dir_name]['max_deriv_order'][1], self.llm_pool.max_deriv_orders['max_deriv_x'])
        return (max_t, max_x)

    def fit(self):
        self.initialize_epde_search_obj()
        if self.use_init_population:
            self.initialize_population()

        terms_max_num = max(epde_params[self._dir_name]['equation_terms_max_number'], self.llm_pool.terms_max_num)
        factors_max_num = max(epde_params[self._dir_name]['equation_factors_max_number'], self.llm_pool.factors_max_num)

        i = 0
        clean_parsed_out(self._dir_name)
        run_eq_info = []
        while i < self.__max_iter:
            start = time.time()

            # epde/interface/interface.py line 743
            self.epde_search_obj.fit(data=self.u, max_deriv_order=self.__get_max_deriv_order(),
                                     equation_terms_max_number=terms_max_num,
                                     equation_factors_max_number=factors_max_num,
                                     eq_sparsity_interval=epde_params[self._dir_name]['eq_sparsity_interval'],
                                     additional_tokens=self.__get_additional_tokens(),
                                     data_fun_pow=max(epde_params[self._dir_name]['data_fun_pow'],
                                                      self.llm_pool.max_deriv_pow['data_fun_pow']),
                                     deriv_fun_pow=max(1, self.llm_pool.max_deriv_pow['deriv_fun_pow']),
                                     population=self.population,
                                     fourier_layers=epde_params[self._dir_name]['fourier_layers'])
            end = time.time()
            self.epde_search_obj.equations(only_print=True, only_str=False, num=epde_params[self._dir_name]['num'])
            res = self.epde_search_obj.equations(only_print=False, only_str=False, num=epde_params[self._dir_name]['num'])
            iter_info = evaluate_fronts(res, self._dir_name, end-start, i)
            # run_eq_info += iter_info

            front_r = FrontReranker(iter_info)
            run_eq_info.append(front_r.select_best())

            print('Overall time is, s:', end-start)
            print(f'Iterations processed: {i + 1}/{self.__max_iter}\n')
            i += 1
        return run_eq_info

    def __get_additional_tokens(self):
        if len(self.__additional_classes) == 0 and epde_params[self._dir_name]['additional_tokens'] is None:
            return []
        else:
            if epde_params[self._dir_name]['additional_tokens'] is None:
                return self.__additional_classes
            else:
                self.__additional_classes.append(epde_params[self._dir_name]['additional_tokens'])
                return self.__additional_classes

    def initialize_epde_search_obj(self):
        if self.epde_search_obj is None:
            self.epde_search_obj = get_epde_search_obj(self.grids, self._dir_name, self._device)

    def initialize_population(self):
        self.population = []
        self.epde_search_obj.create_pool(data=self.u,
                                         max_deriv_order=self.__get_max_deriv_order(),
                                         additional_tokens=self.__get_additional_tokens(),
                                         data_fun_pow=max(epde_params[self._dir_name]['data_fun_pow'],
                                                          self.llm_pool.max_deriv_pow['data_fun_pow']),
                                         deriv_fun_pow=max(1, self.llm_pool.max_deriv_pow['deriv_fun_pow']),
                                         derivs=None)

        for i, eq_u in enumerate(self.__eq_epde_str):
            soeqs = translate_equation(eq_u, pool=self.epde_search_obj.pool, all_vars=['u', ])
            self.population.append(soeqs)
