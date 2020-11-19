from doubleml import DoubleMLPLR
import asyncio
import json
import numpy as np
from sklearn.utils import check_X_y

from .double_ml_aws_lambda import DoubleMLLambda


class DoubleMLPLRServerless(DoubleMLPLR):
    def __init__(self,
                 obj_dml_lambda,
                 obj_dml_data,
                 ml_g,
                 ml_m,
                 n_folds=5,
                 n_rep=1,
                 score='partialling out',
                 dml_procedure='dml2',
                 draw_sample_splitting=True,
                 apply_cross_fitting=True):
        super().__init__(obj_dml_data,
                         ml_g,
                         ml_m,
                         n_folds,
                         n_rep,
                         score,
                         dml_procedure,
                         draw_sample_splitting,
                         apply_cross_fitting)
        assert isinstance(obj_dml_lambda, DoubleMLLambda)
        self._dml_lambda = obj_dml_lambda

    # this method overwrites DoubleML.fit() to implement the fit via aws lambda
    def fit(self, n_jobs_cv=None, keep_scores=True):

        assert self._dml_data.n_treat == 1
        self._i_treat = 0

        # ml estimation of nuisance models and computation of score elements
        psi_a, psi_b = self._ml_nuisance_and_score_elements(self.smpls, n_jobs_cv)
        self._psi_a[:, :, self._i_treat] = psi_a
        self._psi_b[:, :, self._i_treat] = psi_b

        for i_rep in range(self.n_rep):
            self._i_rep = i_rep

            # estimate the causal parameter
            self._all_coef[self._i_treat, self._i_rep] = self._est_causal_pars()

            # compute score (depends on estimated causal parameter)
            self._compute_score()

            # compute standard errors for causal parameter
            self._all_se[self._i_treat, self._i_rep] = self._se_causal_pars()

        # aggregated parameter estimates and standard errors from repeated cross-fitting
        self._agg_cross_fit()

        if not keep_scores:
            self._clean_scores()

        return self

    def _ml_nuisance_and_score_elements(self, smpls, n_jobs_cv):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y)
        x, d = check_X_y(x, self._dml_data.d)

        payload = {
            'bucket': self._dml_data.bucket,
            'file_key': self._dml_data.file_key,
        }

        payload_ml_g = payload.copy()
        payload_ml_m = payload.copy()

        payload_ml_g['learner'] = 'ml_g'
        payload_ml_g['learner_repr'] = self.learner['ml_g'].__repr__()
        payload_ml_g['y_col'] = self._dml_data.y_col
        payload_ml_g['x_cols'] = self._dml_data.x_cols

        payload_ml_m['learner'] = 'ml_m'
        payload_ml_m['learner_repr'] = self.learner['ml_m'].__repr__()
        payload_ml_m['y_col'] = self._dml_data.d_cols[0]
        payload_ml_m['x_cols'] = self._dml_data.x_cols

        payloads = list()

        for i_rep in range(self.n_rep):
            for i_fold, (train_index, test_index) in enumerate(smpls[i_rep]):
                this_payload = payload_ml_g.copy()
                this_payload['i_rep'] = i_rep
                this_payload['i_fold'] = i_fold
                this_payload['train_ids'] = train_index.tolist()
                this_payload['test_ids'] = test_index.tolist()
                payloads.append(this_payload)

                this_payload = payload_ml_m.copy()
                this_payload['i_rep'] = i_rep
                this_payload['i_fold'] = i_fold
                this_payload['train_ids'] = train_index.tolist()
                this_payload['test_ids'] = test_index.tolist()
                payloads.append(this_payload)

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._dml_lambda.invoke(payloads))

        g_hat = np.full((self._dml_data.n_obs, self.n_rep), np.nan)
        m_hat = np.full((self._dml_data.n_obs, self.n_rep), np.nan)

        for this_res in results:
            res_dict = json.loads(this_res)
            assert res_dict['statusCode'] == 200
            test_index = self.smpls[res_dict['i_rep']][res_dict['i_fold']][1]
            if res_dict['learner'] == 'ml_g':
                g_hat[test_index, res_dict['i_rep']] = res_dict['preds']
            else:
                assert res_dict['learner'] == 'ml_m'
                m_hat[test_index, res_dict['i_rep']] = res_dict['preds']

        psi_a = np.full((self._dml_data.n_obs, self.n_rep), np.nan)
        psi_b = np.full((self._dml_data.n_obs, self.n_rep), np.nan)

        for i_rep in range(self.n_rep):
            # compute residuals
            u_hat = y - g_hat[:, i_rep]
            v_hat = d - m_hat[:, i_rep]
            v_hatd = np.multiply(v_hat, d)

            score = self.score
            self._check_score(score)
            if isinstance(self.score, str):
                if score == 'IV-type':
                    psi_a[:, i_rep] = -v_hatd
                else:
                    assert score == 'partialling out'
                    psi_a[:, i_rep] = -np.multiply(v_hat, v_hat)
                psi_b[:, i_rep] = np.multiply(v_hat, u_hat)
            else:
                assert callable(self.score)
                psi_a[:, i_rep], psi_b[:, i_rep] = self.score(y, d, g_hat[:, i_rep], m_hat[:, i_rep], smpls[i_rep])

        return psi_a, psi_b
