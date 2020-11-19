import json
import boto3
from joblib import Parallel, delayed


class DoubleMLLambda:
    def __init__(self,
                 lambda_function_name,
                 aws_region):
        self._lambda_function_name = lambda_function_name
        self._aws_region = aws_region
        self.lambda_client = boto3.client('lambda', region_name=self.aws_region)

    @property
    def aws_region(self):
        return self._aws_region

    @property
    def lambda_function_name(self):
        return self._lambda_function_name

    def invoke(self, payloads):
        n_lambdas = len(payloads)
        parallel = Parallel(n_jobs=n_lambdas, prefer='threads')
        results = parallel(delayed(self.__invoke_single_lambda)(payload) for payload in payloads)
        return results

    def __invoke_single_lambda(self, payload):
        # print(f'Invoking {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')
        response = self.lambda_client.invoke(
            FunctionName=self.lambda_function_name,
            InvocationType='RequestResponse',
            LogType='Tail',
            Payload=json.dumps(payload),
        )
        result = response['Payload'].read()
        # print(f'Finished {payload["learner"]} {payload["i_rep"]} {payload["i_fold"]}')

        return result
