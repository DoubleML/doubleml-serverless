import pandas as pd
import asyncio
import aiobotocore
import json

from .lambda_functions.cv_predict import lambda_cv_predict
from ._helper import _extract_lambda_metrics


class DoubleMLLambda:
    def __init__(self,
                 lambda_function_name,
                 aws_region):
        self._lambda_function_name = lambda_function_name
        self._aws_region = aws_region
        self.aws_lambda_detailed_metrics = pd.DataFrame(columns=['RequestId', 'Duration', 'Billed Duration',
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
        metrics['Total Billed Duration (Seconds)'] = df['Billed Duration'].sum() / 1000
        if metrics['Requests'] > 0:
            metrics['Memory Size (MB; last request)'] = df['Memory Size'].iloc[-1]
        metrics['Max Memory Used (MB)'] = df['Max Memory Used'].max()
        metrics['Avg Max Memory Used (MB)'] = df['Max Memory Used'].mean()
        return metrics

    def invoke_lambdas(self, payloads):
        if self.lambda_function_name == 'local':
            assert self.aws_region == 'local'
            # this callable option is just for local testing
            context = dict()
            results = []
            for this_payload in payloads:
                xx = json.dumps(this_payload)
                yy = json.loads(xx)
                this_res = lambda_cv_predict(yy, context)
                results.append(json.dumps(this_res))
        else:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self.__invoke_aws_lambdas(payloads))

            df_lambda_metrics = _extract_lambda_metrics(results)
            self.aws_lambda_detailed_metrics = self.aws_lambda_detailed_metrics.append(df_lambda_metrics)
        return results

    async def __invoke_aws_lambdas(self, payloads):
        session = aiobotocore.get_session()
        tasks = []
        for this_payload in payloads:
            tasks.append(self.__invoke_single_aws_lambda(session, this_payload))
        results = await asyncio.gather(*tasks)
        return results

    async def __invoke_single_aws_lambda(self, session, payload):
        async with session.create_client('lambda', region_name=self.aws_region) as lambda_client:
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

