from doubleml import DoubleMLData


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
        self._file_key = file_key

    @property
    def bucket(self):
        return self._bucket

    @property
    def file_key(self):
        return self._file_key
