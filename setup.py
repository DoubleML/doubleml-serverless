from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

#PROJECT_URLS = {
#    'Bug Tracker': 'https://github.com/DoubleML/doubleml-serverless/issues',
#    'Source Code': 'https://github.com/DoubleML/doubleml-serverless'
#}

setup(
    name='DoubleML-Serverless',
    version='0.1.dev0',
    author='Kurz, M. S.',
    maintainer='Malte S. Kurz',
    maintainer_email='malte.simon.kurz@uni-hamburg.de',
    description='Double Machine Learning with Serverless Scaling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DoubleML/doubleml-serverless',
    packages=find_packages(exclude=['aws_lambda_app*']),
    install_requires=[
        'DoubleML>=0.1.2',
        'joblib',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'statsmodels',
        'aiobotocore==1.1.2',
        'boto3==1.14.44',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
