from doubleml import DoubleMLIRM
import numpy as np
from sklearn.utils import check_X_y

from ._helper import _get_cond_smpls

from .double_ml_aws_lambda import DoubleMLLambda
from ._helper import _attach_learner, _attach_smpls


class DoubleMLIRMServerless(DoubleMLIRM, DoubleMLLambda):
    def __init__(self,
                 lambda_function_name,
                 aws_region,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='ATE',
                 dml_procedure='dml2',
                 trimming_rule='truncate',
                 trimming_threshold=1e-12,
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        DoubleMLIRM.__init__(self,
                             obj_dml_data,
                             ml_g,
                             ml_m,
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

    def _ml_nuisance_aws_lambda(self, cv_params):
        assert self._dml_data.n_treat == 1
        self._i_treat = 0

        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)
        # get train indices for d == 0 and d == 1
        smpls_d0, smpls_d1 = _get_cond_smpls(self.smpls, d)

        payload = self._dml_data.get_payload()

        payload_ml_g0 = payload.copy()
        payload_ml_g1 = payload.copy()
        payload_ml_m = payload.copy()

        _attach_learner(payload_ml_g0,
                        'ml_g0', self.learner['ml_g'],
                        self._dml_data.y_col, self._dml_data.x_cols)

        if (self.score == 'ATE') | callable(self.score):
            _attach_learner(payload_ml_g1,
                            'ml_g1', self.learner['ml_g'],
                            self._dml_data.y_col, self._dml_data.x_cols)

        _attach_learner(payload_ml_m,
                        'ml_m', self.learner['ml_m'],
                        self._dml_data.d_cols[0], self._dml_data.x_cols,
                        method='predict_proba')
        if (self.score == 'ATE') | callable(self.score):
            all_payloads = [payload_ml_g0, payload_ml_g1, payload_ml_m]
            all_smpls = [smpls_d0, smpls_d1, self.smpls]
        else:
            all_payloads = [payload_ml_g0, payload_ml_m]
            all_smpls = [smpls_d0, self.smpls]

        payloads = _attach_smpls(all_payloads,
                                 all_smpls,
                                 self.n_folds,
                                 self.n_rep,
                                 self._dml_data.n_obs,
                                 cv_params['n_lambdas_cv'],
                                 [True, True, False],
                                 cv_params['seed'])

        preds = self.invoke_lambdas(payloads, self.smpls, self.params_names,
                                    self._dml_data.n_obs, self.n_rep,
                                    cv_params['n_lambdas_cv'])

        for i_rep in range(self.n_rep):
            # compute score elements
            self._psi_a[:, i_rep, self._i_treat], self._psi_b[:, i_rep, self._i_treat] = \
                self._score_elements(y, d,
                                     preds['ml_g0'][:, i_rep],
                                     preds['ml_g1'][:, i_rep],
                                     preds['ml_m'][:, i_rep],
                                     self.smpls[i_rep])

        return
