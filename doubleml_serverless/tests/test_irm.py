import numpy as np
import pandas as pd
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
import doubleml_serverless as dml_lambda
from helper_local_lambda_calls import DoubleMLIRMServerlessLocal

from doubleml_serverless.tests.helper_general import get_n_datasets

# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params=range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[[LogisticRegression(solver='lbfgs', max_iter=250),
                         LinearRegression()],
                        [RandomForestClassifier(max_depth=2, n_estimators=10),
                         RandomForestRegressor(max_depth=2, n_estimators=10)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module")
def dml_irm_fixture(generate_data_irm, idx, learner, score, dml_procedure, trimming_threshold):
    boot_methods = ['normal']
    n_folds = 4
    n_rep_boot = 502

    # collect data
    data = generate_data_irm[idx]

    # to simulate lambda calls we have dumps to json, so we simulate it here to prevent differences due to inaccuracies
    data = pd.read_json(data.to_json(orient='columns'), orient='columns')

    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner[1])
    ml_m = clone(learner[0])

    np.random.seed(3141)
    dml_data_json = dml_lambda.DoubleMLDataJson(data, 'y', ['d'], x_cols)
    dml_irm_lambda = DoubleMLIRMServerlessLocal('local', 'local',
                                                dml_data_json,
                                                ml_g, ml_m,
                                                n_folds,
                                                score=score,
                                                dml_procedure=dml_procedure)

    dml_irm_lambda.fit_aws_lambda()

    np.random.seed(3141)
    dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_irm = dml.DoubleMLIRM(dml_data,
                              ml_g, ml_m,
                              n_folds,
                              score=score,
                              dml_procedure=dml_procedure)

    dml_irm.fit()

    res_dict = {'coef': dml_irm.coef,
                'coef_lambda': dml_irm_lambda.coef,
                'se': dml_irm.se,
                'se_lambda': dml_irm_lambda.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_irm.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_irm_lambda.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_coef' + bootstrap] = dml_irm.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_irm.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_lambda'] = dml_irm_lambda.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_lambda'] = dml_irm_lambda.boot_t_stat

    return res_dict


def test_dml_irm_coef(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['coef'],
                        dml_irm_fixture['coef_lambda'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_se(dml_irm_fixture):
    assert math.isclose(dml_irm_fixture['se'],
                        dml_irm_fixture['se_lambda'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_boot(dml_irm_fixture):
    for bootstrap in dml_irm_fixture['boot_methods']:
        assert np.allclose(dml_irm_fixture['boot_coef' + bootstrap],
                           dml_irm_fixture['boot_coef' + bootstrap + '_lambda'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_irm_fixture['boot_t_stat' + bootstrap],
                           dml_irm_fixture['boot_t_stat' + bootstrap + '_lambda'],
                           rtol=1e-9, atol=1e-4)


@pytest.fixture(scope="module")
def dml_irm_scaling_fixture(generate_data_irm, idx, learner, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 4
    n_rep_boot = 502

    # collect data
    data = generate_data_irm[idx]

    # to simulate lambda calls we have dumps to json, so we simulate it here to prevent differences due to inaccuracies
    data = pd.read_json(data.to_json(orient='columns'), orient='columns')

    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner[1])
    ml_m = clone(learner[0])

    dml_data_json = dml_lambda.DoubleMLDataJson(data, 'y', ['d'], x_cols)

    np.random.seed(3141)
    dml_irm_folds = DoubleMLIRMServerlessLocal('local', 'local',
                                               dml_data_json,
                                               ml_g, ml_m,
                                               n_folds,
                                               score=score,
                                               dml_procedure=dml_procedure)

    dml_irm_folds.fit_aws_lambda('n_folds * n_rep')

    np.random.seed(3141)
    dml_irm_reps = DoubleMLIRMServerlessLocal('local', 'local',
                                              dml_data_json,
                                              ml_g, ml_m,
                                              n_folds,
                                              score=score,
                                              dml_procedure=dml_procedure)

    dml_irm_reps.fit_aws_lambda('n_rep')

    res_dict = {'coef_folds': dml_irm_folds.coef,
                'coef_reps': dml_irm_reps.coef,
                'se_folds': dml_irm_folds.se,
                'se_reps': dml_irm_reps.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        dml_irm_folds.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        np.random.seed(3141)
        dml_irm_reps.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)

        res_dict['boot_coef' + bootstrap + '_folds'] = dml_irm_folds.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_folds'] = dml_irm_folds.boot_t_stat
        res_dict['boot_coef' + bootstrap + '_reps'] = dml_irm_reps.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_reps'] = dml_irm_reps.boot_t_stat

    return res_dict


def test_dml_irm_scaling_coef(dml_irm_scaling_fixture):
    assert math.isclose(dml_irm_scaling_fixture['coef_folds'],
                        dml_irm_scaling_fixture['coef_reps'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_scaling_se(dml_irm_scaling_fixture):
    assert math.isclose(dml_irm_scaling_fixture['se_folds'],
                        dml_irm_scaling_fixture['se_reps'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_scaling_boot(dml_irm_scaling_fixture):
    for bootstrap in dml_irm_scaling_fixture['boot_methods']:
        assert np.allclose(dml_irm_scaling_fixture['boot_coef' + bootstrap + '_folds'],
                           dml_irm_scaling_fixture['boot_coef' + bootstrap + '_reps'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_irm_scaling_fixture['boot_t_stat' + bootstrap + '_folds'],
                           dml_irm_scaling_fixture['boot_t_stat' + bootstrap + '_reps'],
                           rtol=1e-9, atol=1e-4)

