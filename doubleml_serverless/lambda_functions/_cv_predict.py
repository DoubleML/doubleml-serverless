import json
import boto3

import pandas as pd
from sklearn.linear_model import *
from sklearn.ensemble import *


def lambda_cv_predict(event, context):
    # Get variables from event
    bucket = event.get('bucket')
    key = event.get('file_key')
    lrn_repr = event.get('learner_repr')
    y_col = event.get('y_col')
    x_cols = event.get('x_cols')
    train_ids = event.get('train_ids')
    test_ids = event.get('test_ids')

    learner_name = event.get('learner')
    i_rep = event.get('i_rep')
    i_fold = event.get('i_fold')

    # Set client to get file from S3
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket,
                                    Key=key)
    csv_file = response["Body"]

    # Load csv as a Pandas Dataframe
    df = pd.read_csv(csv_file)

    y = df.loc[:, y_col].values
    x = df.loc[:, x_cols].values

    # create and fit learner
    learner = eval(lrn_repr)
    learner.fit(x[train_ids], y[train_ids])
    score_val = learner.score(x[test_ids], y[test_ids])
    preds = learner.predict(x[test_ids])

    return {
        'statusCode': 200,
        'message': 'Success!',
        'score': score_val,
        'preds': preds.tolist(),
        'learner': learner_name,
        'i_rep': i_rep,
        'i_fold': i_fold
    }
