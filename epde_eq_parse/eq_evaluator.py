from epde_eq_parse.eq_parser import get_terms, get_terms_with_coeffs, check_schema, save_equation, resolve_ambiguity
from epde_eq_parse.schemas import schemas
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import os


PARENT_PATH = Path(os.path.dirname(__file__)).parent
class EqInfo(object):
    # EqInfo(self.terms_with_coeffs, loss, mae, shd, runtime, iter_num, eq_eval.is_correct_schema)
    def __init__(self, terms_with_coeffs, obj_val, mae, mae_norm, shd, runtime, iter_num, is_correct=False):
        self.terms_with_coeffs = terms_with_coeffs
        self.obj_val = obj_val
        self.mae = mae
        self.mae_norm = mae_norm
        self.shd = shd
        self.iter_num = iter_num
        self.runtime = runtime
        self.is_correct = is_correct


class EqEvaluator(object):
    def __init__(self, dir_name, terms_with_coeffs):
        self.dir_name = dir_name
        self.terms_with_coeffs = terms_with_coeffs
        self.is_correct_schema = check_schema(schemas[dir_name]['schema'], terms_with_coeffs)

        correct_cfs_set = schemas[dir_name]['params']
        idx = self.get_correct_coeffs_idx(correct_cfs_set)
        self.correct_coeffs = correct_cfs_set[idx]

    def get_correct_coeffs_idx(self, correct_cfs_set):
        coeff_idx, min_diff = 0, 1000000
        for i, coeff_set in enumerate(correct_cfs_set):
            coeff_difference = 0.0
            overall_key_set = set(coeff_set.keys()).union(set(self.terms_with_coeffs.keys()))
            for key in overall_key_set:
                coeff_difference += np.fabs(coeff_set.get(key, 0.0) - self.terms_with_coeffs.get(key, 0.0))

            if coeff_difference < min_diff:
                min_diff = coeff_difference
                coeff_idx = i
        return coeff_idx

    def eval_mae(self, eval_incorrect_eq=False):
        if self.is_correct_schema:
            mae1, mae2 = 0.0, 0.0
            for key in self.terms_with_coeffs.keys():
                mae1 += np.fabs(self.correct_coeffs.get(key, 0.0) - self.terms_with_coeffs[key])
                mae2 += np.fabs(-self.correct_coeffs.get(key, 0.0) - self.terms_with_coeffs[key])
            mae = min(mae1, mae2)
            return mae / len(self.terms_with_coeffs)
        elif eval_incorrect_eq:
            mae = 0.
            for key in self.terms_with_coeffs.keys():
                mae += np.fabs(self.terms_with_coeffs.get(key, 0.0) - self.correct_coeffs.get(key, 0.0))
            return mae / len(self.terms_with_coeffs) if self.terms_with_coeffs['C'] > 0.000000001 else (
                   mae / (len(self.terms_with_coeffs) - 1))
        return None

    def eval_shd(self):
        correct_coeffs_set = set(self.correct_coeffs.keys())
        terms_with_coeffs_set = set(self.terms_with_coeffs.keys())

        delete_wrong_set = terms_with_coeffs_set.difference(correct_coeffs_set)
        add_correct_set = correct_coeffs_set.difference(terms_with_coeffs_set)
        shd = len(add_correct_set) + len(delete_wrong_set)
        return shd

    def eval_mae_norm(self, eval_incorrect_eq=False):
        if set(self.terms_with_coeffs.keys()) == set(self.correct_coeffs.keys()):
            mae1, mae2 = 0.0, 0.0
            for key in self.terms_with_coeffs.keys():
                if self.correct_coeffs[key] != 0:
                    mae1 += np.fabs(self.correct_coeffs[key] - self.terms_with_coeffs[key]) / abs(self.correct_coeffs[key])
                    mae2 += np.fabs(-self.correct_coeffs[key] - self.terms_with_coeffs[key]) / abs(self.correct_coeffs[key])
            mae = min(mae1, mae2)
            return mae / len(self.terms_with_coeffs)

        return None

class FrontReranker(object):
    def __init__(self, iter_info: list[EqInfo]):
        self.iter_info = iter_info
        self.best_info = None

    def select_best(self, metric):
        if metric == "mae":
            if len(self.iter_info) != 0:
                min_mae, min_idx = 10000., 0
                for i, eq_info in enumerate(self.iter_info):
                    if eq_info.mae < min_mae:
                        min_mae = eq_info.mae
                        min_idx = i

                self.best_info = self.iter_info[min_idx]
                return self.best_info
            else:
                return EqInfo(None, None, None, None, None, None, None)

        elif metric == "shd":
            if len(self.iter_info) != 0:
                min_shd, min_idx = 10000., 0
                for i, eq_info in enumerate(self.iter_info):
                    if eq_info.shd < min_shd:
                        min_shd = eq_info.shd
                        min_idx = i

                self.best_info = self.iter_info[min_idx]
                return self.best_info
            else:
                return EqInfo(None, None, None, None, None, None, None)
        else:
            return EqInfo(None, None, None, None, None, None, None)

class EqReranker(object):
    def __init__(self, run_eq_info: list[EqInfo], dir_name: str):
        self.run_eq_info = run_eq_info
        self.dir_name = dir_name
        self.best_run_inf = []

    def select_best(self, metric):
        if metric == "mae":
            min_mae, min_idx = 10000., 0
            iter_idx = 0
            for i, eq_info in enumerate(self.run_eq_info):

                if eq_info.iter_num != iter_idx:
                    self.best_run_inf.append(self.run_eq_info[min_idx])
                    min_mae = 10000.
                    iter_idx += 1

                if eq_info.mae < min_mae:
                    min_mae = eq_info.mae
                    min_idx = i

            self.best_run_inf.append(self.run_eq_info[min_idx])
            return self.best_run_inf

        elif metric == "shd":
            min_shd, min_idx = 10000., 0
            iter_idx = 0
            for i, eq_info in enumerate(self.run_eq_info):

                if eq_info.iter_num != iter_idx:
                    self.best_run_inf.append(self.run_eq_info[min_idx])
                    min_shd = 10000.
                    iter_idx += 1

                if eq_info.shd < min_shd:
                    min_shd = eq_info.shd
                    min_idx = i
            try:
                self.best_run_inf.append(self.run_eq_info[min_idx])
            except:
                print("No optimal equation found")
            return self.best_run_inf
        else:
            return None

    def to_csv(self, package="epde_eq_parse", experiment_info="test"):
        header = ['mae', 'mae_norm', 'shd', 'runtime']
        file_path = os.path.join(PARENT_PATH, package, "metrics", f'{self.dir_name}_metrics_{experiment_info}.csv',)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for eq_info in self.best_run_inf:
                writer.writerow([eq_info.mae, eq_info.mae_norm, eq_info.shd, eq_info.runtime])


# class ResExporter(object):
#     def __init__(self, best_run_inf):
#         self.best_run_inf = best_run_inf
#
#     def export_mae(self):

def evaluate_fronts(pareto_fronts, dir_name, runtime, iter_num):
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
                mae_norm = eq_eval.eval_mae_norm()
                eq_info = EqInfo(terms_with_coeffs, soeq.obj_fun, mae, mae_norm, shd, runtime, iter_num, True)
                # shd, time, iter_num)
                iter_eq_info.append(eq_info)
                save_equation(eq_info, dir_name, iter_num)
            elif dir_name == "burg_sindy" and soeq.obj_fun[0] < 0.026:
                print("Some unknown equation with low obj_fun found")
                eq_info = EqInfo(terms_with_coeffs, soeq.obj_fun, 1000000, 1000000, runtime, iter_num, True)
                iter_eq_info.append(eq_info)
                save_equation(eq_info, dir_name, iter_num)
    return iter_eq_info


class DummySoeq(object):
    def __init__(self, text_form, obj_fun):
        self.text_form = text_form
        self.obj_fun = obj_fun


if __name__ == '__main__':
    # -0.17705425481602444 * d^3u/dx1^3{power: 1.0} + -0.16377446251841735 * du/dx0{power: 1.0} + 0.0 * du/dx0{power: 1.0} * du/dx1{power: 1.0} + -2.057181974238461e-06 = du/dx1{power: 1.0} * u{power: 1.0}
    # -6.099944218365907 * du/dx1 * u + 0.0 * d^3u/dx1^3 * du/dx0 + -1.0793047739940058 * d^3u/dx1^3 + -1.1845818829292476e-05 = du/dx0
    dir_name = 'kdv_sindy'
    iter_num = 0
    runtime = 10

    str1 = '-6.099944218365907 * du/dx1{power: 1.0} * u{power: 1.0} + 0.0 * d^3u/dx1^3{power: 1.0} * du/dx0{power: 1.0} + -1.0793047739940058{power: 1.0} * d^3u/dx1^3{power: 1.0} + -1.1845818829292476e-05 = du/dx0{power: 1.0}'
    str2 = '-0.17705425481602444 * d^3u/dx1^3{power: 1.0} + -0.16377446251841735 * du/dx0{power: 1.0} + 0.0 * du/dx0{power: 1.0} * du/dx1{power: 1.0} + -2.057181974238461e-06 = du/dx1{power: 1.0} * u{power: 1.0}'
    pareto_fronts = [[DummySoeq(str1, 1.0), DummySoeq(str2, 1.0)]]

    p1 = np.fabs(-6.099944218365907 + 6.) + np.fabs(-1.0793047739940058 + 1.) + np.fabs(-1.1845818829292476e-05)
    p1 /= 4

    p2 = 0.17705425481602444 - 1/6 + np.fabs(0.16377446251841735 - 1/6) + 2.057181974238461e-06
    p2 /= 4
    inf = evaluate_fronts(pareto_fronts, dir_name, runtime, iter_num)
    print()
