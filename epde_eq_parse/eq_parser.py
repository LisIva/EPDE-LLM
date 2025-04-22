import re
from epde_eq_parse.schemas import schemas
import numpy as np
import pickle
from pathlib import Path
import shutil
import os

PARENT_PATH = Path(os.path.dirname(__file__)).parent
def check_schema(schema, terms_coeffs):
    keys = set(list(terms_coeffs.keys()))
    return schema.issubset(keys)


def get_terms_with_coeffs(terms, right):
    C_exists = False
    terms_coeffs = {}
    for term in terms:
        idx = term.find(' * ')
        pure_term = term[idx + 3:]
        if idx != -1:
            coeff = float(term[:idx])
            if np.fabs(coeff) > 0.00000001:
                terms_coeffs[pure_term] = coeff
        else:
            C_exists = True
            terms_coeffs['C'] = float(term)

    if not C_exists:
        terms_coeffs['C'] = 0.0

    terms_coeffs[right] = -1.0
    return terms_coeffs


def get_terms(eq_str):
    eq_str = eq_str[:eq_str.find("\n")]
    left, right = eq_str.split(' = ')

    terms = left.split('} + ')
    remove_power_pattern = r'\{power: \d+\.\d+\}?'
    for i in range(len(terms)):
        terms[i] = re.sub(remove_power_pattern, '', terms[i])
    right = re.sub(remove_power_pattern, '', right)
    return terms, right


def save_equation(eq_info, dir_name, iter_num=0):

    def get_last_file_num(dir_path):
        folder = Path(dir_path)
        folder.mkdir(parents=True, exist_ok=True)
        return len(list(folder.iterdir()))

    full_dir = os.path.join(PARENT_PATH, "epde_eq_parse", "parsed_eqs", dir_name, f'iter_{iter_num}')
    num = get_last_file_num(full_dir)
    file_dir = os.path.join(full_dir, f'eq_{num}.pickle')
    with open(file_dir, 'wb') as file:
        pickle.dump(eq_info, file)


def clean_parsed_out(dir_name):
    dir_path = os.path.join(PARENT_PATH, "epde_eq_parse", "parsed_eqs", dir_name)
    folder = Path(dir_path)

    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def resolve_ambiguity(terms_with_coeffs: dict):
    for key in terms_with_coeffs.keys():
        if key == 'du/dx1 * u':
            coeff = terms_with_coeffs[key]
            terms_with_coeffs.pop(key)
            terms_with_coeffs['u * du/dx1'] = coeff
            break


# Structural Hamming distance - количество слагаемых, которые нужно убрать из уравнеиня + те, которые надо добавить
if __name__ == '__main__':
    cr1 = {'12': 1, '2fg': 2}
    set11 = set(cr1.keys())
    correct_coeffs_set = {'u', 'u_x * u', 'C', 'u_xx', 'u_tt'}
    terms_with_coeffs_set = {'u', 'u_x * u', 'C', 'u_xx * u_xx', 'u_tt'}

    delete_wrong_set = terms_with_coeffs_set.difference(correct_coeffs_set)
    add_correct_set = correct_coeffs_set.difference(terms_with_coeffs_set)
    shd = len(add_correct_set) + len(delete_wrong_set)

    eq_str = '1.0 * d^2u/dx1^2{power: 1.0} + 0.040768734249692226 * du/dx1{power: 1.0} * u{power: 1.0} + 0.0346445115714672 * d^3u/dx1^3{power: 2.0} * cos(t + 1){power: 1.0} + 0.233 * sin(1 + 3){power: 3.0} + 0.35678891 = d^2u/dx0^2{power: 1.0}'
    terms, right = get_terms(eq_str)
    terms_with_coeffs = get_terms_with_coeffs(terms, right)
    resolve_ambiguity(terms_with_coeffs)
    is_correct_schema = check_schema(schemas['wave']['schema'], terms_with_coeffs)
    # eq_eval = EqEvaluator('wave', terms_with_coeffs)
    # save_equation(terms_with_coeffs, 'wave', (0.111, 5.), iter_num=1)
    # clean_parsed_out('wave')
    print()

    '''
    Load pickle:
    
    file_name = f'parsed_eqs/wave/iter_1/eq_3.pickle'
    with open(file_name, 'rb') as file:
        object_file = pickle.load(file)
    '''
