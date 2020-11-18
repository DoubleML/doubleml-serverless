import asyncio
import json
import aioboto3


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

    async def invoke(self, payloads):
        async with aioboto3.client('lambda', region_name=self.aws_region) as lambda_client:
            tasks = []
            for this_payload in payloads:
                tasks.append(self.__invoke_single_lambda(lambda_client, this_payload))
            results = await asyncio.gather(*tasks)
        return results

    async def __invoke_single_lambda(self, lambda_client, payload):
        print(f'Invoking {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
        response = await lambda_client.invoke(
            FunctionName=self.lambda_function_name,
            InvocationType='RequestResponse',
            LogType='Tail',
            Payload=json.dumps(payload),
        )
        result = await response['Payload'].read()
        print(f'Finished {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')

        return result
