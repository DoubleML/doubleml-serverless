# DoubleML-Serverless - Distributed Double Machine Learning with a Serverless Architecture

This repo contains a prototype implementation **DoubleML-Serverless** of distributed double machine learning with a serverless infrastructure
using [AWS Lambda](https://aws.amazon.com/lambda).
A detailed discussion of this prototype can be found in the paper "Distributed Double Machine Learning with a  Serverless Architecture" (Kurz, 2021).
**DoubleML-Serverless** is an extension for serverless cloud computing of the Python package **DoubleML**.
**DoubleML** is available via PyPI [https://pypi.org/project/DoubleML](https://pypi.org/project/DoubleML) and on GitHub [https://github.com/DoubleML/doubleml-for-py](https://github.com/DoubleML/doubleml-for-py).
Also see [https://docs.doubleml.org](https://docs.doubleml.org) for a detailed documentation and user guide for the **DoubleML** package.

## Getting started

### Installation of DoubleML-Serverless

To install download the latest source code from GitHub via
```
git clone git@github.com:DoubleML/doubleml-serverless.git
cd doubleml-serverless
```

Then build the package from source using pip in the editable mode.

```
pip install --editable .
```

Alternatively to the installation from source, released versions of the DoubleML-Serverless package in form of
.whl files can be obtained from [GitHub Releases](https://github.com/DoubleML/doubleml-serverless/releases).
After downloading the wheel, the package can be installed with pip (replace `XXX` with the downloaded package version).
```
pip install -U DoubleML-Serverless-XXX-py3-none-any.whl
```

### Deploy the corresponding serverless app to AWS Lambda using AWS SAM

To use AWS Lambda for estimating double machine learning models, a deployment in your AWS account is necessary.
The corresponding serverless application consists of the following components:

* A AWS Lambda function called `LambdaCVPredict` (the source code is taken from this repository [https://github.com/DoubleML/doubleml-serverless/blob/master/aws_lambda_app/lambda_functions/cv_predict.py](https://github.com/DoubleML/doubleml-serverless/blob/master/aws_lambda_app/lambda_functions/cv_predict.py)).
* A layer providing the Python libraries `scikit-learn`, `pandas` and `numpy` together with their dependencies.
* An S3 bucket for the data transfer (can be optionally generated, or an existing bucket is used).
* A role for the execution of the lambda function `LambdaCVPredict` which consists of the AWS-managed `AWSLambdaBasicExecutionRole` policy plus read access to the S3 bucket for data transfer.


There are two options for deployment:

1. A version of DoubleML-Serverless is available in the AWS Serverless Application Repository. It can be deployed by clicking on the `Deploy` button.

2. The second option for deployment is based on AWS Serverless Application Model (AWS SAM).

2.1 Setup the AWS SAM CLI as described here: [https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started.html](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started.html)

2.2 To deploy the application use the following commands (for more information see [https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html))

```
cd aws_lambda_app
sam build
sam deploy --guided
```

### Estimating a partially linear regression model with double machine learning and serverless scaling using AWS Lambda



## References
Kurz, M.S. 2020. "Distributed Double Machine Learning with a  Serverless Architecture". Unpublished Working Paper.
