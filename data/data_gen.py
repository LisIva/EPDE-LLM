import os
import numpy as np
from epde.interface.interface import EpdeSearch
import pandas as pd
from pathlib import Path
from scipy.io import loadmat


PARENT_PATH = Path(os.path.dirname(__file__)).parent

def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data

def get_data(dataset):
    if dataset == "burg":
        path = "data_burg"
        path_full = os.path.join(Path().absolute().parent, path, "burgers_sln_100.csv")
        df = pd.read_csv(path_full, header=None)

        u = df.values
        u = np.transpose(u)
        x = np.linspace(-1000, 0, 101)
        t = np.linspace(0, 1, 101)
        grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2) , axis = 2)

    elif dataset == "burg_sindy":
        dir_burg = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "data_burg")
        burg = loadmat(os.path.join(dir_burg, "burgers.mat"))
        t = np.ravel(burg['t'])
        x = np.ravel(burg['x'])
        u = np.real(burg['usol'])
        u = np.transpose(u)
        grids = np.meshgrid(t, x, indexing='ij')

    elif dataset == "kdv_sindy":
        path_full = os.path.join(Path().absolute().parent, "data_kdv", "kdv.mat")
        kdV = loadmat(path_full)
        t = np.ravel(kdV['t'])
        x = np.ravel(kdV['x'])
        u = np.real(kdV['usol'])
        u = np.transpose(u)
        grids = np.meshgrid(t, x, indexing='ij')

    elif dataset == "wave":
        base_path = Path().absolute().parent
        path_full = os.path.join(base_path, "data_wave", "wave_sln_100.csv")
        df = pd.read_csv(path_full, header=None)
        u = df.values
        u = np.transpose(u)
        t = np.linspace(0, 1, 101)
        x = np.linspace(0, 1, 101)
        grids = np.meshgrid(t, x, indexing='ij')

    return grids, u, x, t


def gen_derivs(noise_level, dataset, i=None):
    if i is not None:
        full_path = Path(os.path.join(Path().absolute().parent, f"data/noise_level_{noise_level}/", f"{dataset}/", f"{i}/"))
    else:
        full_path = Path(
            os.path.join(Path().absolute().parent, f"data/noise_level_{noise_level}/", f"{dataset}/"))
    full_path.mkdir(parents=True, exist_ok=True)

    t_deriv_order = 2
    x_deriv_order = 3

    grid, data, x, t = get_data(dataset)
    noised_data = noise_data(data, noise_level)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=20,
                                 coordinate_tensors=grid, device='cuda')

    if noise_level == 0:
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                         preprocessor_kwargs={})
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={"use_smoothing": True})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ],
                                max_deriv_order=(t_deriv_order, x_deriv_order),
                                additional_tokens=[])
    derivs = epde_search_obj._derivatives

    np.save(os.path.join(full_path, "u"), noised_data)
    np.save(os.path.join(full_path, "x"), x)
    np.save(os.path.join(full_path, "t"), t)

    if noise_level == 0:
        for i in range(t_deriv_order):
            if i == 0:
                np.save(os.path.join(full_path, "du_dx1"), derivs['u'][:, i])
            else:
                np.save(os.path.join(full_path, f"d^{i+1}u_dx1^{i+1}"), derivs['u'][:, i])

        for i in range(x_deriv_order):
            if i == 0:
                np.save(os.path.join(full_path, "du_dx2"), derivs['u'][:, i + t_deriv_order])
            else:
                np.save(os.path.join(full_path, f"d^{i+1}u_dx2^{i+1}"), derivs['u'][:, i + t_deriv_order])
    else:
        for i in range(t_deriv_order):
            if i == 0:
                np.save(os.path.join(full_path, "du_dx1"), derivs['u'][:, i].reshape(data.shape))
            else:
                np.save(os.path.join(full_path, f"d^{i+1}u_dx1^{i+1}"), derivs['u'][:, i].reshape(data.shape))

        for i in range(x_deriv_order):
            if i == 0:
                np.save(os.path.join(full_path, "du_dx2"), derivs['u'][:, i + t_deriv_order].reshape(data.shape))
            else:
                np.save(os.path.join(full_path, f"d^{i+1}u_dx2^{i+1}"), derivs['u'][:, i + t_deriv_order].reshape(data.shape))

if __name__ == "__main__":
    noise_level = 10
    single = True
    iters = 30
    datasets = ["burg", "burg_sindy", "kdv_sindy", "wave"]

    for dataset in datasets:
        if noise_level == 0 or single:
            gen_derivs(noise_level, dataset, None)
        else:
            for i in range(iters):
                gen_derivs(noise_level, dataset, i)




