import numpy as np
import pandas as pd

import pytest

from doubleml.datasets import make_plr_turrell2018, make_irm_data, make_iivm_data, make_pliv_CHS2015

from doubleml_serverless.tests.helper_general import get_n_datasets

# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='session',
                params=[(500, 15), ])
def generate_data_plr(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_plr_turrell2018(N, p, theta, return_type=pd.DataFrame)
        datasets.append(data)

    return datasets


@pytest.fixture(scope='session',
                params=[(500, 15), ])
def generate_data_pliv(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_pliv_CHS2015(n_obs=N, dim_x=p, alpha=theta, dim_z=1, return_type=pd.DataFrame)
        datasets.append(data)

    return datasets


@pytest.fixture(scope='session',
                params=[(500, 15), ])
def generate_data_irm(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_irm_data(N, p, theta, return_type=pd.DataFrame)
        datasets.append(data)

    return datasets


@pytest.fixture(scope='session',
                params=[(500, 15), ])
def generate_data_iivm(request):
    N_p = request.param
    np.random.seed(1111)
    # setting parameters
    N = N_p[0]
    p = N_p[1]
    theta = 0.5
    gamma_z = 0.4

    # generating data
    datasets = []
    for i in range(n_datasets):
        data = make_iivm_data(N, p, theta, gamma_z, return_type=pd.DataFrame)
        datasets.append(data)

    return datasets
