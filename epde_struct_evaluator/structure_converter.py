import re
import numpy as np

class StructConverter(object):
    def __init__(self, eq_str, params, left_side):
        self.eq_key = eq_str
        self.eq_str = eq_str
        self.params = params
        self.left_side = left_side
        self.terms_dict = None

    def convert(self):
        self.eq_str = self.replace_with_params()
        self.eq_str = self.replace_x_and_t()
        # self.eq_str = self.replace_number_formatting()
        self.terms_dict = self.get_dict_terms()
        self.resolve_ambiguity()
        return self.terms_dict

    # def replace_number_formatting(self):
    #     return re.sub(r':\.\d+f\b', ' ', self.eq_str)

    def resolve_ambiguity(self):
        for key in self.terms_dict.keys():
            if key == 'du/dx1 * u' or key == 'du/dx1*u':
                coeff = self.terms_dict[key]
                self.terms_dict.pop(key)
                self.terms_dict['u * du/dx1'] = coeff
                break

    def replace_x_and_t(self):
        eq_str = self.eq_str.replace('x', 'x1')
        return eq_str.replace('t', 'x0')

    def get_dict_terms(self):
        left_right = self.eq_str.split(' = ')
        if len(left_right) == 2:
            terms = left_right[-1].split(' + ')
        else:
            terms = left_right[0].split(' + ')

        terms_dict = {}
        for term in terms:
            idx = term.find('*')
            coef = float(term[:idx])
            term = term[idx+1:].strip()
            terms_dict[term] = coef

        terms_dict[self.left_side] = -1.
        terms_dict['C'] = 0.
        return terms_dict

    def replace_with_params(self):
        pattern = r"c\[(\d+)\]"

        if bool(re.search(pattern, self.eq_str)):
            def replace_c_with_params(match):
                # Extract the index from the match
                idx_str = match.group(1)
                idx = int(idx_str)
                # Check if the index is within the bounds of the params vector
                if 0 <= idx < len(self.params):
                    return str(self.params[idx])
                else:
                    raise IndexError("The len of 'c' and 'params' do not align")

            return re.sub(pattern, replace_c_with_params, self.eq_str)

        elif bool(re.search(r"\bc\b", self.eq_str)):
            # Replace standalone c
            return re.sub(r'\bc\b', str(self.params[0]), self.eq_str)


if __name__ == '__main__':
    s1 = 'c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x'
    params = np.array([1.5, 1.22, 5.67])

    sc = StructConverter(s1, params)
    terms = sc.convert()
    print(sc.eq_str)
