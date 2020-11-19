import asyncio
import aiobotocore
import json


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
        session = aiobotocore.get_session()
        tasks = []
        for this_payload in payloads:
            tasks.append(self.__invoke_single_lambda(session, this_payload))
        results = await asyncio.gather(*tasks)
        return results

    async def __invoke_single_lambda(self, session, payload):
        async with session.create_client('lambda', region_name=self.aws_region) as lambda_client:
            # logging.info(f'Invoking {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
            response = await lambda_client.invoke(
                FunctionName=self.lambda_function_name,
                InvocationType='RequestResponse',
                LogType='Tail',
                Payload=json.dumps(payload),
            )
            # logging.info(f'Done {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
            async with response['Payload'] as stream:
                result = await stream.read()
            # logging.info(f'Finished {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')

        return result
