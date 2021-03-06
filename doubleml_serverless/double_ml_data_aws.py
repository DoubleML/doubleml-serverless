from doubleml import DoubleMLData
import os
import boto3
import pandas as pd


class DoubleMLDataS3(DoubleMLData):
    def __init__(self,
                 bucket,
                 file_key,
                 data,
                 y_col,
                 d_cols,
                 x_cols=None,
                 z_cols=None,
                 use_other_treat_as_covariate=True):
        super().__init__(data,
                         y_col,
                         d_cols,
                         x_cols,
                         z_cols,
                         use_other_treat_as_covariate)
        self._bucket = bucket
        self._file_ending = os.path.splitext(file_key)[1]
        assert self._file_ending in ['.csv']
        self._file_key = file_key

    @property
    def bucket(self):
        return self._bucket

    @property
    def file_key(self):
        return self._file_key

    def get_payload(self):
        payload = {
            'data_backend': 's3',
            'bucket': self.bucket,
            'file_key': self.file_key,
        }
        return payload

    @classmethod
    def from_s3(cls,
                bucket,
                file_key,
                y_col,
                d_cols,
                x_cols=None,
                z_cols=None,
                use_other_treat_as_covariate=True):
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket,
                                        Key=file_key)
        file = response["Body"]
        file_ending = os.path.splitext(file_key)[1]
        assert file_ending in ['.csv']
        # load csv as a pd.DataFrame
        data = pd.read_csv(file)

        return cls(bucket, file_key, data, y_col, d_cols, x_cols, z_cols, use_other_treat_as_covariate)

    def store_and_upload_to_s3(self):
        # load csv as a pd.DataFrame
        file_name = os.path.split(self.file_key)[1]
        self.data.to_csv(file_name)
        s3_client = boto3.client('s3')
        response = s3_client.upload_file(Filename=file_name,
                                         Bucket=self.bucket,
                                         Key=self.file_key)
        return response


class DoubleMLDataJson(DoubleMLData):
    def __init__(self,
                 data,
                 y_col,
                 d_cols,
                 x_cols=None,
                 z_cols=None,
                 use_other_treat_as_covariate=True):
        super().__init__(data,
                         y_col,
                         d_cols,
                         x_cols,
                         z_cols,
                         use_other_treat_as_covariate)
        self._data_json = data.to_json(orient="columns")

    @property
    def data_json(self):
        return self._data_json

    def get_payload(self):
        payload = {
            'data_backend': 'json',
            'data': self.data_json
        }
        return payload
