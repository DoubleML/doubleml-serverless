import asyncio
import aiobotocore
from botocore import UNSIGNED
from botocore.config import Config
import json

import numpy as np
from numpy.random import MT19937, RandomState
import pandas as pd
from sklearn.linear_model import *
from sklearn.ensemble import *

from doubleml_serverless._helper import _extract_preds, _extract_lambda_metrics
from doubleml_serverless import DoubleMLPLRServerless, DoubleMLPLIVServerless,\
    DoubleMLIRMServerless, DoubleMLIIVMServerless


# this is a duplicate of the deployed lambda code
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
    seed = event.get('seed')
    seed_jumps = event.get('seed_jumps')

    if data_backend == 'json':
        df_json = event.get('data')
        df = pd.read_json(df_json, orient='columns')
    else:
        raise NotImplementedError()

    y = df.loc[:, y_col].values
    x = df.loc[:, x_cols].values

    # create and fit learner

    learner = eval(lrn_repr)
    if scaling == 'n_folds * n_rep':
        if seed is not None:
            learner.set_params(random_state=RandomState(MT19937(seed).jumped(seed_jumps)))
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
            for idx, test_index in enumerate(test_ids):
                if seed is not None:
                    learner.set_params(random_state=RandomState(MT19937(seed).jumped(seed_jumps + idx)))
                learner.fit(np.delete(x, test_index, axis=0), np.delete(y, test_index))
                if pred_method == 'predict':
                    preds[test_index] = learner.predict(x[test_index])
                else:
                    assert pred_method == 'predict_proba'
                    preds[test_index] = learner.predict_proba(x[test_index])[:, 1]
        else:
            for idx, (train_index, test_index) in enumerate(zip(train_ids, test_ids)):
                if seed is not None:
                    learner.set_params(random_state=RandomState(MT19937(seed).jumped(seed_jumps + idx)))
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


def pseudo_invoke_lambdas_locally(payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
    # this callable option is just for local testing
    context = dict()
    results = []
    for this_payload in payloads:
        xx = json.dumps(this_payload)
        yy = json.loads(xx)
        this_res = dict()
        this_res['payload'] = json.dumps(lambda_cv_predict(yy, context))
        results.append(this_res)
    preds, requests = _extract_preds(results, smpls, params_names,
                                     n_obs, n_rep, n_jobs_cv)
    return preds


class DoubleMLPLRServerlessLocal(DoubleMLPLRServerless):
    def invoke_lambdas(self, payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
        assert self.lambda_function_name == 'local'
        assert self.aws_region == 'local'
        preds = pseudo_invoke_lambdas_locally(payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv)
        return preds


class DoubleMLPLIVServerlessLocal(DoubleMLPLIVServerless):
    def invoke_lambdas(self, payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
        assert self.lambda_function_name == 'local'
        assert self.aws_region == 'local'
        preds = pseudo_invoke_lambdas_locally(payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv)
        return preds


class DoubleMLIRMServerlessLocal(DoubleMLIRMServerless):
    def invoke_lambdas(self, payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
        assert self.lambda_function_name == 'local'
        assert self.aws_region == 'local'
        preds = pseudo_invoke_lambdas_locally(payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv)
        return preds


class DoubleMLIIVMServerlessLocal(DoubleMLIIVMServerless):
    def invoke_lambdas(self, payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
        assert self.lambda_function_name == 'local'
        assert self.aws_region == 'local'
        preds = pseudo_invoke_lambdas_locally(payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv)
        return preds


# if sam local start-lambda allows larger payloads, we could use the following implementation for testing
# see https://github.com/aws/aws-sam-cli/issues/188
# class DoubleMLPLRServerlessLocal(DoubleMLPLRServerless):
#     def invoke_lambdas(self, payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
#         loop = asyncio.get_event_loop()
#         results = loop.run_until_complete(self.__invoke_aws_lambdas_locally(payloads))
#         preds, requests = _extract_preds(results, smpls, params_names,
#                                          n_obs, n_rep, n_jobs_cv)
#         return preds
#
#     async def __invoke_aws_lambdas_locally(self, payloads):
#         session = aiobotocore.get_session()
#         tasks = []
#         for this_payload in payloads:
#             tasks.append(self.__invoke_single_aws_lambda_locally(session, this_payload))
#         results = await asyncio.gather(*tasks)
#         return results
#
#     async def __invoke_single_aws_lambda_locally(self, session, payload):
#         async with session.create_client('lambda',
#                                          endpoint_url='http://127.0.0.1:3001',
#                                          use_ssl=False,
#                                          verify=False,
#                                          config=Config(signature_version=UNSIGNED,
#                                                        read_timeout=0,
#                                                        retries={'max_attempts': 0})
#                                          ) as lambda_client:
#             # print(f'Invoking {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
#             response = await lambda_client.invoke(
#                 FunctionName=self.lambda_function_name,
#                 InvocationType='RequestResponse',
#                 LogType='None',
#                 Payload=json.dumps(payload),
#             )
#             # print(f'Done {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
#             res = dict()
#             async with response['Payload'] as stream:
#                 res['payload'] = await stream.read()
#             # res['log'] = response['LogResult']
#             # print(f'Finished {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
#
#         return res


