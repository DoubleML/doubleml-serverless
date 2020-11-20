from doubleml import DoubleMLPLR
import asyncio
import json
import numpy as np
from sklearn.utils import check_X_y

from .double_ml_aws_lambda import DoubleMLLambda
from .double_ml_data_aws import DoubleMLDataS3, DoubleMLDataJson
from ._helper import _attach_learner, _attach_smpls, _extract_preds


class DoubleMLPLRServerless(DoubleMLPLR, DoubleMLLambda):
    def __init__(self,
                 lambda_function_name,
                 aws_region,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        DoubleMLPLR.__init__(self,
                             obj_dml_data,
                             ml_g,
                             ml_m,
                             n_folds,
                             n_rep,
                             score,
                             dml_procedure,
                             draw_sample_splitting,
                             apply_cross_fitting)
        DoubleMLLambda.__init__(self,
                                lambda_function_name,
                                aws_region)

    # this method overwrites DoubleML.fit() to implement the fit via aws lambda
    def fit(self, n_jobs_cv='n_folds * n_rep', keep_scores=True):
        """
        Parameters
        ----------
        n_jobs_cv : str

        keep_scores : bool
        """

        if (not isinstance(n_jobs_cv, str)) | (n_jobs_cv not in ['n_folds * n_rep', 'n_rep']):
            raise ValueError('n_jobs_cv must be "n_folds * n_rep" or "n_rep"'
                             f' got {str(n_jobs_cv)}')

        assert self._dml_data.n_treat == 1
        self._i_treat = 0

        # ml estimation of nuisance models and computation of score elements
        psi_a, psi_b = self._ml_nuisance_and_score_elements(self.smpls, n_jobs_cv)
        self._psi_a[:, :, self._i_treat] = psi_a
        self._psi_b[:, :, self._i_treat] = psi_b

        self._est_causal_pars_and_se()

        if not keep_scores:
            self._clean_scores()

        return self

    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)

        payload = self._dml_data.get_payload()

        payload_ml_g = payload.copy()
        payload_ml_m = payload.copy()

        _attach_learner(payload_ml_g,
                        'ml_g', self.learner['ml_g'],
                        self._dml_data.y_col, self._dml_data.x_cols)

        _attach_learner(payload_ml_m,
                        'ml_m', self.learner['ml_m'],
                        self._dml_data.d_cols[0], self._dml_data.x_cols)

        payloads = _attach_smpls([payload_ml_g, payload_ml_m],
                                 smpls, self._dml_data.n_obs,
                                 n_jobs_cv)

        results = self.invoke_lambdas(payloads)

        preds = _extract_preds(results, smpls, self.params_names,
                               self._dml_data.n_obs, self.n_rep,
                               n_jobs_cv)

        psi_a = np.full((self._dml_data.n_obs, self.n_rep), np.nan)
        psi_b = np.full((self._dml_data.n_obs, self.n_rep), np.nan)

        for i_rep in range(self.n_rep):
            # compute score elements
            psi_a[:, i_rep], psi_b[:, i_rep] = self._score_elements(y, d,
                                                                    preds['ml_g'][:, i_rep],
                                                                    preds['ml_m'][:, i_rep],
                                                                    smpls)

        return psi_a, psi_b
