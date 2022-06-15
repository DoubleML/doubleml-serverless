import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml
import doubleml_serverless as dml_lambda

from doubleml_serverless.tests.helper_local_lambda_calls import DoubleMLPLRServerlessLocal
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
def dml_plr_fixture(generate_data_plr, idx, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 4
    n_rep_boot = 502

    # collect data
    data = generate_data_plr[idx]

    # to simulate lambda calls we have dumps to json, so we simulate it here to prevent differences due to inaccuracies
    data = pd.read_json(data.to_json(orient='columns'), orient='columns')

    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    dml_data_json = dml_lambda.DoubleMLDataJson(data, 'y', ['d'], x_cols)
    dml_plr_lambda = DoubleMLPLRServerlessLocal('local', 'local',
                                                dml_data_json,
                                                ml_l, ml_m,
                                                n_folds=n_folds,
                                                score=score,
                                                dml_procedure=dml_procedure)

    dml_plr_lambda.fit_aws_lambda()

    np.random.seed(3141)
    dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr = dml.DoubleMLPLR(dml_data,
                              ml_l, ml_m,
                              n_folds=n_folds,
                              score=score,
                              dml_procedure=dml_procedure)

    dml_plr.fit()

    res_dict = {'coef': dml_plr.coef,
                'coef_lambda': dml_plr_lambda.coef,
                'se': dml_plr.se,
                'se_lambda': dml_plr_lambda.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_plr.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_plr_lambda.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_coef' + bootstrap] = dml_plr.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_lambda'] = dml_plr_lambda.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_lambda'] = dml_plr_lambda.boot_t_stat

    return res_dict


def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_lambda'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_lambda'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_coef' + bootstrap],
                           dml_plr_fixture['boot_coef' + bootstrap + '_lambda'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_lambda'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_plr_scaling_fixture(generate_data_plr, idx, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 4
    n_rep_boot = 502

    # collect data
    data = generate_data_plr[idx]

    # to simulate lambda calls we have dumps to json, so we simulate it here to prevent differences due to inaccuracies
    data = pd.read_json(data.to_json(orient='columns'), orient='columns')

    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_l = clone(learner)
    ml_m = clone(learner)

    dml_data_json = dml_lambda.DoubleMLDataJson(data, 'y', ['d'], x_cols)

    np.random.seed(3141)
    dml_plr_folds = DoubleMLPLRServerlessLocal('local', 'local',
                                               dml_data_json,
                                               ml_l, ml_m,
                                               n_folds=n_folds,
                                               score=score,
                                               dml_procedure=dml_procedure)

    dml_plr_folds.fit_aws_lambda('n_folds * n_rep')

    np.random.seed(3141)
    dml_plr_reps = DoubleMLPLRServerlessLocal('local', 'local',
                                              dml_data_json,
                                              ml_l, ml_m,
                                              n_folds=n_folds,
                                              score=score,
                                              dml_procedure=dml_procedure)

    dml_plr_reps.fit_aws_lambda('n_rep')

    res_dict = {'coef_folds': dml_plr_folds.coef,
                'coef_reps': dml_plr_reps.coef,
                'se_folds': dml_plr_folds.se,
                'se_reps': dml_plr_reps.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_plr_folds.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_plr_reps.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_coef' + bootstrap + '_folds'] = dml_plr_folds.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_folds'] = dml_plr_folds.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_reps'] = dml_plr_reps.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_reps'] = dml_plr_reps.boot_t_stat

    return res_dict


def test_dml_plr_scaling_coef(dml_plr_scaling_fixture):
    assert math.isclose(dml_plr_scaling_fixture['coef_folds'],
                        dml_plr_scaling_fixture['coef_reps'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_scaling_se(dml_plr_scaling_fixture):
    assert math.isclose(dml_plr_scaling_fixture['se_folds'],
                        dml_plr_scaling_fixture['se_reps'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_scaling_boot(dml_plr_scaling_fixture):
    for bootstrap in dml_plr_scaling_fixture['boot_methods']:
        assert np.allclose(dml_plr_scaling_fixture['boot_coef' + bootstrap + '_folds'],
                           dml_plr_scaling_fixture['boot_coef' + bootstrap + '_reps'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_scaling_fixture['boot_t_stat' + bootstrap + '_folds'],
                           dml_plr_scaling_fixture['boot_t_stat' + bootstrap + '_reps'],
                           rtol=1e-9, atol=1e-4)

