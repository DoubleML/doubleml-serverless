import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml
import doubleml_serverless as dml_lambda

from doubleml_serverless.tests.helper_local_lambda_calls import DoubleMLPLIVServerlessLocal
from doubleml_serverless.tests.helper_general import get_n_datasets

# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params=range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[RandomForestRegressor(max_depth=2, n_estimators=10),
                        LinearRegression(),
                        Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_pliv_fixture(generate_data_pliv, idx, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 4
    n_rep_boot = 502

    # collect data
    data = generate_data_pliv[idx]

    # to simulate lambda calls we have dumps to json, so we simulate it here to prevent differences due to inaccuracies
    data = pd.read_json(data.to_json(orient='columns'), orient='columns')

    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    dml_data_json = dml_lambda.DoubleMLDataJson(data, 'y', ['d'], x_cols, 'Z1')
    dml_pliv_lambda = DoubleMLPLIVServerlessLocal('local', 'local',
                                                  dml_data_json,
                                                  ml_l, ml_m, ml_r,
                                                  n_folds=n_folds,
                                                  score=score,
                                                  dml_procedure=dml_procedure)

    dml_pliv_lambda.fit_aws_lambda()

    np.random.seed(3141)
    dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols, 'Z1')
    dml_pliv = dml.DoubleMLPLIV(dml_data,
                                ml_l, ml_m, ml_r,
                                n_folds=n_folds,
                                score=score,
                                dml_procedure=dml_procedure)

    dml_pliv.fit()

    res_dict = {'coef': dml_pliv.coef,
                'coef_lambda': dml_pliv_lambda.coef,
                'se': dml_pliv.se,
                'se_lambda': dml_pliv_lambda.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_pliv.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_pliv_lambda.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_coef' + bootstrap] = dml_pliv.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_pliv.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_lambda'] = dml_pliv_lambda.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_lambda'] = dml_pliv_lambda.boot_t_stat

    return res_dict


def test_dml_pliv_coef(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['coef'],
                        dml_pliv_fixture['coef_lambda'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_se(dml_pliv_fixture):
    assert math.isclose(dml_pliv_fixture['se'],
                        dml_pliv_fixture['se_lambda'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_boot(dml_pliv_fixture):
    for bootstrap in dml_pliv_fixture['boot_methods']:
        assert np.allclose(dml_pliv_fixture['boot_coef' + bootstrap],
                           dml_pliv_fixture['boot_coef' + bootstrap + '_lambda'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_pliv_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_fixture['boot_t_stat' + bootstrap + '_lambda'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_pliv_scaling_fixture(generate_data_pliv, idx, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 4
    n_rep_boot = 502

    # collect data
    data = generate_data_pliv[idx]

    # to simulate lambda calls we have dumps to json, so we simulate it here to prevent differences due to inaccuracies
    data = pd.read_json(data.to_json(orient='columns'), orient='columns')

    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    dml_data_json = dml_lambda.DoubleMLDataJson(data, 'y', ['d'], x_cols, 'Z1')

    np.random.seed(3141)
    dml_pliv_folds = DoubleMLPLIVServerlessLocal('local', 'local',
                                                 dml_data_json,
                                                 ml_l, ml_m, ml_r,
                                                 n_folds=n_folds,
                                                 score=score,
                                                 dml_procedure=dml_procedure)

    dml_pliv_folds.fit_aws_lambda('n_folds * n_rep')

    np.random.seed(3141)
    dml_pliv_reps = DoubleMLPLIVServerlessLocal('local', 'local',
                                                dml_data_json,
                                                ml_l, ml_m, ml_r,
                                                n_folds=n_folds,
                                                score=score,
                                                dml_procedure=dml_procedure)

    dml_pliv_reps.fit_aws_lambda('n_rep')

    res_dict = {'coef_folds': dml_pliv_folds.coef,
                'coef_reps': dml_pliv_reps.coef,
                'se_folds': dml_pliv_folds.se,
                'se_reps': dml_pliv_reps.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_pliv_folds.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_pliv_reps.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_coef' + bootstrap + '_folds'] = dml_pliv_folds.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_folds'] = dml_pliv_folds.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_reps'] = dml_pliv_reps.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_reps'] = dml_pliv_reps.boot_t_stat

    return res_dict


def test_dml_pliv_scaling_coef(dml_pliv_scaling_fixture):
    assert math.isclose(dml_pliv_scaling_fixture['coef_folds'],
                        dml_pliv_scaling_fixture['coef_reps'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_scaling_se(dml_pliv_scaling_fixture):
    assert math.isclose(dml_pliv_scaling_fixture['se_folds'],
                        dml_pliv_scaling_fixture['se_reps'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_scaling_boot(dml_pliv_scaling_fixture):
    for bootstrap in dml_pliv_scaling_fixture['boot_methods']:
        assert np.allclose(dml_pliv_scaling_fixture['boot_coef' + bootstrap + '_folds'],
                           dml_pliv_scaling_fixture['boot_coef' + bootstrap + '_reps'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_pliv_scaling_fixture['boot_t_stat' + bootstrap + '_folds'],
                           dml_pliv_scaling_fixture['boot_t_stat' + bootstrap + '_reps'],
                           rtol=1e-9, atol=1e-4)

