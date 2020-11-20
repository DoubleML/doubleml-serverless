import asyncio
import aiobotocore
import json

from .lambda_functions.cv_predict import lambda_cv_predict


class DoubleMLLambda:
    def __init__(self,
                 lambda_function_name,
                 aws_region):
        self._lambda_function_name = lambda_function_name
        self._aws_region = aws_region

    @property
    def aws_region(self):
        return self._aws_region

    @property
    def lambda_function_name(self):
        return self._lambda_function_name

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
            async with response['Payload'] as stream:
                result = await stream.read()
            # print(f'Finished {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')

        return result

