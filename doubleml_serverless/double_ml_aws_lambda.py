import pandas as pd
import asyncio
from aiobotocore.session import get_session
from aiobotocore.config import AioConfig
import json
import time

from abc import ABC, abstractmethod

from ._helper import _extract_preds, _extract_lambda_metrics


class DoubleMLLambda(ABC):

    def __init__(self,
                 lambda_function_name,
                 aws_region):
        self._lambda_function_name = lambda_function_name
        self._aws_region = aws_region
        self._response_time_lambda = None
        self.aws_lambda_detailed_metrics = pd.DataFrame(columns=['learner', 'i_rep', 'i_fold',
                                                                 'RequestId', 'Duration', 'Billed Duration',
                                                                 'Memory Size', 'Max Memory Used', 'Init Duration',
                                                                 'Billed Duration GBSeconds'])

    @property
    def aws_region(self):
        return self._aws_region

    @property
    def lambda_function_name(self):
        return self._lambda_function_name

    @property
    def aws_lambda_metrics(self):
        df = self.aws_lambda_detailed_metrics
        metrics = pd.Series()
        metrics['Requests'] = df.shape[0]
        metrics['Total Billed Duration (GBSeconds)'] = df['Billed Duration GBSeconds'].sum()
        metrics['Total Duration (Seconds)'] = df['Duration'].sum() / 1000
        metrics['Avg Duration (Seconds)'] = df['Duration'].mean() / 1000
        metrics['Total Billed Duration (Seconds)'] = df['Billed Duration'].sum() / 1000
        if metrics['Requests'] > 0:
            metrics['Memory Size (MB; last request)'] = df['Memory Size'].iloc[-1]
        metrics['Max Memory Used (MB)'] = df['Max Memory Used'].max()
        metrics['Avg Max Memory Used (MB)'] = df['Max Memory Used'].mean()
        metrics['Response Time'] = self._response_time_lambda
        return metrics

    @abstractmethod
    def _ml_nuisance_aws_lambda(self, cv_params):
        pass

    @abstractmethod
    def _est_causal_pars_and_se(self):
        pass

    @abstractmethod
    def _clean_scores(self):
        pass

    def fit_aws_lambda(self, n_lambdas_cv='n_folds * n_rep', seed=None, keep_scores=True):
        """
        Parameters
        ----------
        n_lambdas_cv : str

        seed : int or None

        keep_scores : bool
        """
        if (not isinstance(n_lambdas_cv, str)) | (n_lambdas_cv not in ['n_folds * n_rep', 'n_rep']):
            raise ValueError('n_lambdas_cv must be "n_folds * n_rep" or "n_rep"'
                             f' got {str(n_lambdas_cv)}')

        # ml estimation of nuisance models and computation of score elements
        cv_params = {'n_lambdas_cv': n_lambdas_cv,
                     'seed': seed}
        self._ml_nuisance_aws_lambda(cv_params)

        self._est_causal_pars_and_se()

        if not keep_scores:
            self._clean_scores()

        return self

    def invoke_lambdas(self, payloads, smpls, params_names, n_obs, n_rep, n_jobs_cv):
        loop = asyncio.get_event_loop()
        start_time = time.time()
        results = loop.run_until_complete(self.__invoke_aws_lambdas(payloads))
        end_time = time.time()
        self._response_time_lambda = end_time - start_time
        preds, requests = _extract_preds(results, smpls, params_names,
                                         n_obs, n_rep, n_jobs_cv)
        df_lambda_metrics = _extract_lambda_metrics(results)
        self.aws_lambda_detailed_metrics = self.aws_lambda_detailed_metrics.append(
            pd.concat((requests, df_lambda_metrics), axis=1))
        return preds

    async def __invoke_aws_lambdas(self, payloads):
        session = get_session()
        tasks = []
        for this_payload in payloads:
            tasks.append(self.__invoke_single_aws_lambda(session, this_payload))
        results = await asyncio.gather(*tasks)
        return results

    async def __invoke_single_aws_lambda(self, session, payload):
        config = AioConfig({'keepalive_timeout': 1000}, connect_timeout=1000, read_timeout=1000)
        async with session.create_client('lambda', region_name=self.aws_region, config=config) as lambda_client:
            # print(f'Invoking {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
            response = await lambda_client.invoke(
                FunctionName=self.lambda_function_name,
                InvocationType='RequestResponse',
                LogType='Tail',
                Payload=json.dumps(payload),
            )
            # print(f'Done {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
            res = dict()
            async with response['Payload'] as stream:
                res['payload'] = await stream.read()
            res['log'] = response['LogResult']
            # print(f'Finished {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')

        return res
