import pickle

file_name = f'parsed_eqs/burg_sindy/iter_16/eq_0.pickle'
with open(file_name, 'rb') as file:
    object_file = pickle.load(file)

print()