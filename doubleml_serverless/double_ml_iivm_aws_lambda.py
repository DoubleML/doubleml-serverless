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
                 subgroups=None,
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
                              subgroups,
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
        x, z = check_X_y(x, np.ravel(self._dml_data.z))
        x, d = check_X_y(x, self._dml_data.d)
        # get train indices for z == 0 and z == 1
        smpls_z0, smpls_z1 = _get_cond_smpls(self.smpls, z)

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

        all_payloads = [payload_ml_g0, payload_ml_g1, payload_ml_m]
        all_smpls = [smpls_z0, smpls_z1, self.smpls]
        send_train_ids = [True, True, False]
        params_names = ['ml_g0', 'ml_g1', 'ml_m']

        if self.subgroups['always_takers']:
            _attach_learner(payload_ml_r0,
                            'ml_r0', self.learner['ml_r'],
                            self._dml_data.d_cols[0], self._dml_data.x_cols,
                            method='predict_proba')
            all_payloads.append(payload_ml_r0)
            all_smpls.append(smpls_z0)
            send_train_ids.append(True)
            params_names.append('ml_r0')

        if self.subgroups['never_takers']:
            _attach_learner(payload_ml_r1,
                            'ml_r1', self.learner['ml_r'],
                            self._dml_data.d_cols[0], self._dml_data.x_cols,
                            method='predict_proba')
            all_payloads.append(payload_ml_r1)
            all_smpls.append(smpls_z1)
            send_train_ids.append(True)
            params_names.append('ml_r1')

        payloads = _attach_smpls(all_payloads,
                                 all_smpls,
                                 self.n_folds,
                                 self.n_rep,
                                 self._dml_data.n_obs,
                                 cv_params['n_lambdas_cv'],
                                 send_train_ids,
                                 cv_params['seed'])

        preds = self.invoke_lambdas(payloads, self.smpls, params_names,
                                    self._dml_data.n_obs, self.n_rep,
                                    cv_params['n_lambdas_cv'])

        if not self.subgroups['always_takers']:
            preds['ml_r0'] = np.zeros_like(preds['ml_g0'])
        if not self.subgroups['never_takers']:
            preds['ml_r1'] = np.ones_like(preds['ml_g1'])

        for i_rep in range(self.n_rep):
            # compute score elements

            self._psi_a[:, i_rep, self._i_treat], self._psi_b[:, i_rep, self._i_treat] = \
                self._score_elements(y, z, d,
                                     preds['ml_g0'][:, i_rep],
                                     preds['ml_g1'][:, i_rep],
                                     preds['ml_m'][:, i_rep],
                                     preds['ml_r0'][:, i_rep],
                                     preds['ml_r1'][:, i_rep],
                                     self.smpls[i_rep])

        return
