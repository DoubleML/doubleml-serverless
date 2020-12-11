from doubleml import DoubleMLIIVM
import numpy as np
from sklearn.utils import check_X_y

from ._helper import _get_cond_smpls

from .double_ml_aws_lambda import DoubleMLLambda
from ._helper import _attach_learner, _attach_smpls


class DoubleMLIIVMServerless(DoubleMLIIVM, DoubleMLLambda):
    def __init__(self,
                 lambda_function_name,
                 aws_region,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 ml_r,
                 n_folds=5,
                 n_rep=1,
                 score='ATE',
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        DoubleMLIIVM.__init__(self,
                              obj_dml_data,
                              ml_g,
                              ml_m,
                              ml_r,
                              n_folds,
                              n_rep,
                              score,
                              dml_procedure,
                              trimming_rule,
                              trimming_threshold,
                              draw_sample_splitting,
                              apply_cross_fitting)
        DoubleMLLambda.__init__(self,
                                lambda_function_name,
                                aws_region)

    # this method overwrites DoubleML.fit() to implement the fit via aws lambda
    def fit(self, n_jobs_cv='n_folds * n_rep', seed=None, keep_scores=True):
        """
        Parameters
        ----------
        n_jobs_cv : str

        seed : int or None

        keep_scores : bool
        """

        if (not isinstance(n_jobs_cv, str)) | (n_jobs_cv not in ['n_folds * n_rep', 'n_rep']):
            raise ValueError('n_jobs_cv must be "n_folds * n_rep" or "n_rep"'
                             f' got {str(n_jobs_cv)}')

        assert self._dml_data.n_treat == 1
        self._i_treat = 0

        # ml estimation of nuisance models and computation of score elements
        psi_a, psi_b = self._ml_nuisance_and_score_elements(self.smpls, n_jobs_cv, seed)
        self._psi_a[:, :, self._i_treat] = psi_a
        self._psi_b[:, :, self._i_treat] = psi_b

        self._est_causal_pars_and_se()

        if not keep_scores:
            self._clean_scores()

        return self

    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv, seed):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, z = check_X_y(x, np.ravel(self._dml_data.z))
        x, d = check_X_y(x, self._dml_data.d)
        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = _get_cond_smpls(smpls, z)

        payload = self._dml_data.get_payload()

        payload_ml_g0 = payload.copy()
        payload_ml_g1 = payload.copy()
        payload_ml_m = payload.copy()
        payload_ml_r0 = payload.copy()
        payload_ml_r1 = payload.copy()

        _attach_learner(payload_ml_g0,
                        'ml_g0', self.learner['ml_g'],
                        self._dml_data.y_col, self._dml_data.x_cols)

        _attach_learner(payload_ml_g1,
                        'ml_g1', self.learner['ml_g'],
                        self._dml_data.y_col, self._dml_data.x_cols)

        _attach_learner(payload_ml_m,
                        'ml_m', self.learner['ml_m'],
                        self._dml_data.z_cols[0], self._dml_data.x_cols,
                        method='predict_proba')

        _attach_learner(payload_ml_r0,
                        'ml_r0', self.learner['ml_r'],
                        self._dml_data.d_cols[0], self._dml_data.x_cols,
                        method='predict_proba')

        _attach_learner(payload_ml_r1,
                        'ml_r1', self.learner['ml_r'],
                        self._dml_data.d_cols[0], self._dml_data.x_cols,
                        method='predict_proba')

        all_payloads = [payload_ml_g0, payload_ml_g1, payload_ml_m, payload_ml_r0, payload_ml_r1]
        all_smpls = [smpls_z0, smpls_z1, smpls, smpls_z0, smpls_z1]

        payloads = _attach_smpls(all_payloads,
                                 all_smpls,
                                 self.n_folds,
                                 self.n_rep,
                                 self._dml_data.n_obs,
                                 n_jobs_cv,
                                 [True, True, False, True, True],
                                 seed)

        preds = self.invoke_lambdas(payloads, smpls, self.params_names,
                                    self._dml_data.n_obs, self.n_rep,
                                    n_jobs_cv)

        psi_a = np.full((self._dml_data.n_obs, self.n_rep), np.nan)
        psi_b = np.full((self._dml_data.n_obs, self.n_rep), np.nan)

        for i_rep in range(self.n_rep):
            # compute score elements
            psi_a[:, i_rep], psi_b[:, i_rep] = self._score_elements(y, z, d,
                                                                    preds['ml_g0'][:, i_rep],
                                                                    preds['ml_g1'][:, i_rep],
                                                                    preds['ml_m'][:, i_rep],
                                                                    preds['ml_r0'][:, i_rep],
                                                                    preds['ml_r1'][:, i_rep],
                                                                    smpls[i_rep])

        return psi_a, psi_b
