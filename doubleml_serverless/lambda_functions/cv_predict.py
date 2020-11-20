import json
import boto3

import numpy as np
import pandas as pd
from sklearn.linear_model import *
from sklearn.ensemble import *


def lambda_cv_predict(event, context):
    # Get variables from event
    data_backend = event.get('data_backend')
    lrn_repr = event.get('learner_repr')
    pred_method = event.get('pred_method')
    y_col = event.get('y_col')
    x_cols = event.get('x_cols')
    test_ids = event.get('test_ids')
    train_ids = event.get('train_ids')

    learner_name = event.get('learner')
    scaling = event.get('scaling')
    i_rep = event.get('i_rep')
    i_fold = event.get('i_fold')

    if data_backend == 's3':
        # s3 data backend
        bucket = event.get('bucket')
        key = event.get('file_key')
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket,
                                        Key=key)
        csv_file = response["Body"]
        # load csv as a pd.DataFrame
        df = pd.read_csv(csv_file)
    elif data_backend == 'json':
        df_json = event.get('data')
        df = pd.read_json(df_json, orient='columns')
    else:
        raise NotImplementedError()

    y = df.loc[:, y_col].values
    x = df.loc[:, x_cols].values

    # create and fit learner

    learner = eval(lrn_repr)
    if scaling == 'n_folds * n_rep':
        if train_ids is None:
            learner.fit(np.delete(x, test_ids, axis=0), np.delete(y, test_ids))
        else:
            learner.fit(x[train_ids], y[train_ids])
        if pred_method == 'predict':
            preds = learner.predict(x[test_ids])
        else:
            assert pred_method == 'predict_proba'
            preds = learner.predict_proba(x[test_ids])[:, 1]

    else:
        assert scaling == 'n_rep'
        n_obs = x.shape[0]
        preds = np.full(n_obs, np.nan)
        if train_ids is None:
            for test_index in test_ids:
                learner.fit(np.delete(x, test_index, axis=0), np.delete(y, test_index))
                if pred_method == 'predict':
                    preds[test_index] = learner.predict(x[test_index])
                else:
                    assert pred_method == 'predict_proba'
                    preds[test_index] = learner.predict_proba(x[test_index])[:, 1]
        else:
            for (train_index, test_index) in zip(train_ids, test_ids):
                learner.fit(x[train_index], y[train_index])
                if pred_method == 'predict':
                    preds[test_index] = learner.predict(x[test_index])
                else:
                    assert pred_method == 'predict_proba'
                    preds[test_index] = learner.predict_proba(x[test_index])[:, 1]

    return {
        'statusCode': 200,
        'message': 'Success!',
        'preds': preds.tolist(),
        'learner': learner_name,
        'i_rep': i_rep,
        'i_fold': i_fold
    }
