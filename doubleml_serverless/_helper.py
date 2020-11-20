import numpy as np
import json


def _check_inds_are_partition(inds, n_obs):
    if len(inds) != n_obs:
        return False
    hit = np.zeros(n_obs, dtype=bool)
    hit[inds] = True
    if not np.all(hit):
        return False
    return True


def _attach_learner(payload, learner_name, learner, outcome_var, covars):
    payload['learner'] = learner_name
    payload['learner_repr'] = learner.__repr__()
    payload['y_col'] = outcome_var
    payload['x_cols'] = covars
    return


def _attach_smpls(learner_payloads, smpls, n_obs, scaling):
    payloads = list()
    # note that the order of the loops is not optimal but aligned with the DoubleML package to allow reproducibility
    for i_rep, this_smpl in enumerate(smpls):
        for payload_learner in learner_payloads:
            if scaling == 'n_folds * n_rep':
                for i_fold, (train_index, test_index) in enumerate(this_smpl):
                    _check_inds_are_partition(np.concatenate((train_index, test_index)), n_obs)
                    this_payload = payload_learner.copy()
                    this_payload['scaling'] = scaling
                    this_payload['i_rep'] = i_rep
                    this_payload['i_fold'] = i_fold
                    this_payload['test_ids'] = test_index.tolist()

                    payloads.append(this_payload)
            else:
                assert scaling == 'n_rep'
                test_ids = [test_index.tolist() for (_, test_index) in this_smpl]
                this_payload = payload_learner.copy()
                this_payload['scaling'] = scaling
                this_payload['i_rep'] = i_rep
                this_payload['test_ids'] = test_ids

                payloads.append(this_payload)

    return payloads


def _extract_preds(results, smpls, keys, n_obs, n_rep, scaling):
    preds = {key: np.full((n_obs, n_rep), np.nan) for key in keys}

    if scaling == 'n_folds * n_rep':
        for this_res in results:
            res_dict = json.loads(this_res)
            assert res_dict['statusCode'] == 200
            test_index = smpls[res_dict['i_rep']][res_dict['i_fold']][1]
            preds[res_dict['learner']][test_index, res_dict['i_rep']] = res_dict['preds']
    else:
        assert scaling == 'n_rep'
        for this_res in results:
            res_dict = json.loads(this_res)
            assert res_dict['statusCode'] == 200
            preds[res_dict['learner']][:, res_dict['i_rep']] = res_dict['preds']

    return preds
