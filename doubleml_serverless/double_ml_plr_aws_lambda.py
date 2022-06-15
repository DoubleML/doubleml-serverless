from doubleml import DoubleMLPLR
from sklearn.utils import check_X_y

from .double_ml_aws_lambda import DoubleMLLambda
from ._helper import _attach_learner, _attach_smpls


class DoubleMLPLRServerless(DoubleMLPLR, DoubleMLLambda):
    def __init__(self,
                 lambda_function_name,
                 aws_region,
                 obj_dml_data,
                 ml_l,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        DoubleMLPLR.__init__(self,
                             obj_dml_data=obj_dml_data,
                             ml_l=ml_l,
                             ml_m=ml_m,
                             n_folds=n_folds,
                             n_rep=n_rep,
                             score=score,
                             dml_procedure=dml_procedure,
                             draw_sample_splitting=draw_sample_splitting,
                             apply_cross_fitting=apply_cross_fitting)
        DoubleMLLambda.__init__(self,
                                lambda_function_name=lambda_function_name,
                                aws_region=aws_region)

    def _ml_nuisance_aws_lambda(self, cv_params):
        assert self._dml_data.n_treat == 1
        self._i_treat = 0

        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)

        payload = self._dml_data.get_payload()

        payload_ml_l = payload.copy()
        payload_ml_m = payload.copy()

        _attach_learner(payload_ml_l,
                        'ml_l', self.learner['ml_l'],
                        self._dml_data.y_col, self._dml_data.x_cols)

        _attach_learner(payload_ml_m,
                        'ml_m', self.learner['ml_m'],
                        self._dml_data.d_cols[0], self._dml_data.x_cols)

        payloads = _attach_smpls([payload_ml_l, payload_ml_m],
                                 [self.smpls, self.smpls],
                                 self.n_folds,
                                 self.n_rep,
                                 self._dml_data.n_obs,
                                 cv_params['n_lambdas_cv'],
                                 [False, False],
                                 cv_params['seed'])

        preds = self.invoke_lambdas(payloads, self.smpls, self.params_names,
                                    self._dml_data.n_obs, self.n_rep,
                                    cv_params['n_lambdas_cv'])

        for i_rep in range(self.n_rep):
            # compute score elements
            self._psi_a[:, i_rep, self._i_treat], self._psi_b[:, i_rep, self._i_treat] = \
                self._score_elements(y, d,
                                     preds['ml_l'][:, i_rep],
                                     preds['ml_m'][:, i_rep],
                                     None,
                                     self.smpls[i_rep])

        return
