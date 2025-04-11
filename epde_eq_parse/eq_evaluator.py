from epde_eq_parse.eq_parser import get_terms, get_terms_with_coeffs, check_schema, save_equation, resolve_ambiguity
from epde_eq_parse.schemas import schemas
import numpy as np


class EqInfo(object):
    def __init__(self, terms_with_coeffs, obj_val, mae, shd):
        self.terms_with_coeffs = terms_with_coeffs
        self.obj_val = obj_val
        self.mae = mae
        self.shd = shd


class EqEvaluator(object):
    def __init__(self, dir_name, terms_with_coeffs):
        self.dir_name = dir_name
        self.terms_with_coeffs = terms_with_coeffs

        correct_cfs_set = schemas[dir_name]['params']
        idx = self.get_correct_coeffs_idx(correct_cfs_set)
        self.correct_coeffs = correct_cfs_set[idx]

    def get_correct_coeffs_idx(self, correct_cfs_set):
        coeff_idx, min_diff = 0, 1000000
        for i, coeff_set in enumerate(correct_cfs_set):
            coeff_difference = 0.0
            for key in coeff_set.keys():
                coeff_difference += np.fabs(np.fabs(coeff_set[key]) - np.fabs(self.terms_with_coeffs[key]))

            if coeff_difference < min_diff:
                min_diff = coeff_difference
                coeff_idx = i
        return coeff_idx

    def eval_mae(self):
        mae1, mae2 = 0.0, 0.0
        for key in self.terms_with_coeffs.keys():
            mae1 += np.fabs(self.correct_coeffs.get(key, 0.0) - self.terms_with_coeffs[key])
            mae2 += np.fabs(-self.correct_coeffs.get(key, 0.0) - self.terms_with_coeffs[key])
        mae = min(mae1, mae2)
        return mae / len(self.terms_with_coeffs)

    def eval_shd(self):
        correct_coeffs_set = set(self.correct_coeffs.keys())
        terms_with_coeffs_set = set(self.terms_with_coeffs.keys())

        delete_wrong_set = terms_with_coeffs_set.difference(correct_coeffs_set)
        add_correct_set = correct_coeffs_set.difference(terms_with_coeffs_set)
        return len(add_correct_set) + len(delete_wrong_set)


def evaluate_fronts(pareto_fronts, dir_name, iter_num):
    iter_eq_info = []
    for front in pareto_fronts:
        for soeq in front:
            eq_str = soeq.text_form
            terms, right = get_terms(eq_str)
            terms_with_coeffs = get_terms_with_coeffs(terms, right)
            resolve_ambiguity(terms_with_coeffs)
            is_correct_schema = check_schema(schemas[dir_name]['schema'], terms_with_coeffs)
            if is_correct_schema:
                eq_eval = EqEvaluator(dir_name, terms_with_coeffs)
                mae = eq_eval.eval_mae()
                shd = eq_eval.eval_shd()
                eq_info = EqInfo(terms_with_coeffs, soeq.obj_fun, mae, shd)
                iter_eq_info.append(eq_info)
                save_equation(eq_info, dir_name, iter_num)
    return iter_eq_info


