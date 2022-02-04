# DoubleML-Serverless - Distributed Double Machine Learning with a Serverless Architecture <a href="https://docs.doubleml.org"><img src="https://raw.githubusercontent.com/DoubleML/doubleml-for-py/master/doc/logo.png" align="right" width = "120" /></a>

This repo contains a prototype implementation **DoubleML-Serverless** of distributed double machine learning with a serverless infrastructure
using [AWS Lambda](https://aws.amazon.com/lambda).
A detailed discussion of this prototype can be found in the paper ["Distributed Double Machine Learning with a Serverless Architecture" (Kurz, 2021)](https://doi.org/10.1145/3447545.3451181).
DoubleML-Serverless is an extension for serverless cloud computing of the Python package **DoubleML**.
DoubleML is available via PyPI [https://pypi.org/project/DoubleML](https://pypi.org/project/DoubleML) and on GitHub [https://github.com/DoubleML/doubleml-for-py](https://github.com/DoubleML/doubleml-for-py).
The Python package DoubleML was introduced in
"DoubleML - An Object-Oriented Implementation of Double Machine Learning in Python"
([Bach et al., 2022](https://www.jmlr.org/papers/v23/21-0862.html))
and a detailed documentation \& user guide for the package is available at
[https://docs.doubleml.org](https://docs.doubleml.org).

## Getting Started

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

### Deploy the Corresponding Serverless App to AWS Lambda using AWS SAM

To use AWS Lambda for estimating double machine learning models, a deployment in your AWS account is necessary.
The corresponding serverless application consists of the following components:

* A AWS Lambda function called `LambdaCVPredict` (the source code is taken from this repository [https://github.com/DoubleML/doubleml-serverless/blob/master/aws_lambda_app/lambda_functions/cv_predict.py](https://github.com/DoubleML/doubleml-serverless/blob/master/aws_lambda_app/lambda_functions/cv_predict.py)).
* A layer providing the Python libraries `scikit-learn`, `pandas` and `numpy` together with their dependencies.
* An S3 bucket for the data transfer (can be optionally generated, or an existing bucket is used).
* A role for the execution of the lambda function `LambdaCVPredict` which consists of the AWS-managed `AWSLambdaBasicExecutionRole` policy plus read access to the S3 bucket for data transfer.


There are two options for deployment:

1. A version of DoubleML-Serverless is available in the AWS Serverless Application Repository: [https://serverlessrepo.aws.amazon.com/applications/eu-central-1/839779594349/doubleml-serverless](https://serverlessrepo.aws.amazon.com/applications/eu-central-1/839779594349/doubleml-serverless). It can be deployed by clicking on the `Deploy` button.

2. The second option for deployment is based on AWS Serverless Application Model (AWS SAM).

    2.1 Setup the AWS SAM CLI as described here: [https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started.html](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started.html)

    2.2 To deploy the application use the following commands (for more information see [https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html))
    ```
    cd aws_lambda_app
    sam build
    sam deploy --guided
    ```

### Estimating a Partially Linear Regression Model with Double Machine Learning and Serverless Scaling Using AWS Lambda

To demonstrate the functionality of DoubleML-Serverless we revisit the Pennsylvania  Reemployment Bonus experiment
and estimate the effect of provisioning a cash bonus on the unemployment duration as studied in [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097).
This example is also discussed in the accompanying paper to the DoubleML-Serverless package ([Kurz, 2021](https://doi.org/10.1145/3447545.3451181)).

We first load the data using functionalities from the DoubleML package.
```python
from doubleml.datasets import fetch_bonus
df_bonus = fetch_bonus('DataFrame')
```

The class `DoubleMLDataS3` serves as data-backend for DoubleML-Serverless model classes.
It is inherited from the `DoubleML` class `DoubleMLData`.
We initialize an object of the `DoubleMLDataS3` for the bonus data and upload it to the S3 bucket `doubleml-serverless-data` used for the data transfer to AWS Lambda.
```python
from doubleml_serverless import DoubleMLDataS3

dml_data_bonus = DoubleMLDataS3(
    'doubleml-serverless-data', 'bonus_data.csv',
    df_bonus,
    y_col='inuidur1',
    d_cols='tg',
    x_cols=['female', 'black', 'othrace',
       'dep1', 'dep2', 'q2', 'q3',
       'q4', 'q5', 'q6', 'agelt35',
       'agegt54', 'durable', 'lusd', 'husd'])
dml_data_bonus.store_and_upload_to_s3()
```

To estimate the nuisance functions we use a random forest regressor which averages over 500 trees.
We further apply repeated cross-fitting with 5 folds and 100 repetitions/splits.
```python
from doubleml_serverless import DoubleMLPLRServerless
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

ml = RandomForestRegressor(n_estimators = 500)
ml_g = clone(ml)
ml_m = clone(ml)
dml_lambda_plr_bonus = DoubleMLPLRServerless(
    'LambdaCVPredict', 'eu-central-1',
    dml_data_bonus, ml_g, ml_m,
    n_folds=5, n_rep=100)
```

To estimate the model locally we can call `dml_lambda_plr_bonus.fit()`.
Estimation on AWS Lambda is achieved via `dml_lambda_plr_bonus.fit_aws_lambda()`.
Note that you will be charged for all used resources in the AWS account you deployed the serverless application to.
```python
dml_lambda_plr_bonus.fit_aws_lambda()
```

A summary of the estimation result is available via the property `dml_lambda_plr_bonus.summary`.
Some metrics about the estimation on AWS Lambda can be obtained via the property  `dml_lambda_plr_bonus.aws_lambda_metrics`.

## Citation

If you use the DoubleML-Serverless package a citation is highly appreciated:

Kurz, M. S. (2021). Distributed Double Machine Learning with a Serverless Architecture.
In Companion of the ACM/SPEC International Conference on Performance Engineering (ICPE '21).
Association for Computing Machinery, New York, NY, USA, 27–33.
doi:[10.1145/3447545.3451181](https://doi.org/10.1145/3447545.3451181).

Bibtex-entry:

```
@inproceedings{kurz2021DoublemlServerless,
   author = {Kurz, Malte S.},
   title = {Distributed Double Machine Learning with a Serverless Architecture},
   year = {2021},
   isbn = {9781450383318},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   url = {https://doi.org/10.1145/3447545.3451181},
   doi = {10.1145/3447545.3451181},
   abstract = {This paper explores serverless cloud computing for double machine learning. Being based on repeated cross-fitting, double machine learning is particularly well suited to exploit the high level of parallelism achievable with serverless computing. It allows to get fast on-demand estimations without additional cloud maintenance effort. We provide a prototype Python implementation DoubleML-Serverless for the estimation of double machine learning models with the serverless computing platform AWS Lambda and demonstrate its utility with a case study analyzing estimation times and costs.},
   booktitle = {Companion of the ACM/SPEC International Conference on Performance Engineering},
   pages = {27--33},
   numpages = {7},
   keywords = {machine learning, causal machine learning, serverless computing, distributed computing, AWS Lambda, function-as-a-service (FAAS)},
   location = {Virtual Event, France},
   series = {ICPE '21}
}
```

## References

Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M. (2022), DoubleML - An
Object-Oriented Implementation of Double Machine Learning in Python,
Journal of Machine Learning Research, 23(53): 1-6,
[https://www.jmlr.org/papers/v23/21-0862.html](https://www.jmlr.org/papers/v23/21-0862.html).

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018).
Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68.
doi:[10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097).

Kurz, M. S. (2021). Distributed Double Machine Learning with a Serverless Architecture.
In Companion of the ACM/SPEC International Conference on Performance Engineering (ICPE '21).
Association for Computing Machinery, New York, NY, USA, 27–33.
doi:[10.1145/3447545.3451181](https://doi.org/10.1145/3447545.3451181).
