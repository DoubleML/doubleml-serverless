import numpy as np
import pandas as pd

import json
import base64


def _check_inds_are_partition(inds, n_obs):
    if len(inds) != n_obs:
        return False
    hit = np.zeros(n_obs, dtype=bool)
    hit[inds] = True
    if not np.all(hit):
        return False
    return True


def _get_cond_smpls(smpls, bin_var):
    smpls_0 = list()
    smpls_1 = list()
    for this_smpl in smpls:
        smpls_0.append([(np.intersect1d(np.where(bin_var == 0)[0], train), test) for train, test in this_smpl])
        smpls_1.append([(np.intersect1d(np.where(bin_var == 1)[0], train), test) for train, test in this_smpl])
    return smpls_0, smpls_1


def _attach_learner(payload, learner_name, learner, outcome_var, covars, method='predict'):
    payload['learner'] = learner_name
    payload['learner_repr'] = learner.__repr__()
    payload['y_col'] = outcome_var
    payload['x_cols'] = covars
    payload['pred_method'] = method
    return


def _attach_smpls(learner_payloads, smpls, n_rep, n_obs, scaling, send_train_ids):
    payloads = list()
    # note that the order of the loops is not optimal but aligned with the DoubleML package to allow reproducibility
    for i_rep in range(n_rep):
        for i_learner, payload_learner in enumerate(learner_payloads):
            this_smpl = smpls[i_learner][i_rep]
            if scaling == 'n_folds * n_rep':
                for i_fold, (train_index, test_index) in enumerate(this_smpl):
                    this_payload = payload_learner.copy()
                    this_payload['scaling'] = scaling
                    this_payload['i_rep'] = i_rep
                    this_payload['i_fold'] = i_fold
                    if send_train_ids[i_learner]:
                        this_payload['train_ids'] = train_index.tolist()
                    else:
                        assert _check_inds_are_partition(np.concatenate((train_index, test_index)), n_obs)
                    this_payload['test_ids'] = test_index.tolist()

                    payloads.append(this_payload)
            else:
                assert scaling == 'n_rep'
                this_payload = payload_learner.copy()
                this_payload['scaling'] = scaling
                this_payload['i_rep'] = i_rep
                test_ids = [test_index.tolist() for (_, test_index) in this_smpl]
                if send_train_ids[i_learner]:
                    train_ids = [train_index.tolist() for (train_index, _) in this_smpl]
                    this_payload['train_ids'] = train_ids
                else:
                    smpls_are_partitions = [_check_inds_are_partition(np.concatenate((train_index, test_index)), n_obs)
                                            for (train_index, test_index) in this_smpl]
                    assert all(smpls_are_partitions)
                this_payload['test_ids'] = test_ids

                payloads.append(this_payload)

    return payloads


def _extract_preds(results, smpls, keys, n_obs, n_rep, scaling):
    preds = {key: np.full((n_obs, n_rep), np.nan) for key in keys}

    if scaling == 'n_folds * n_rep':
        fields = ['learner', 'i_rep', 'i_fold']
        requests = {key: list() for key in fields}
        for this_res in results:
            res_dict = json.loads(this_res['payload'])
            assert res_dict['statusCode'] == 200
            for key in fields:
                requests[key].append(res_dict[key])
            test_index = smpls[res_dict['i_rep']][res_dict['i_fold']][1]
            preds[res_dict['learner']][test_index, res_dict['i_rep']] = res_dict['preds']
    else:
        assert scaling == 'n_rep'
        fields = ['learner', 'i_rep']
        requests = {key: list() for key in fields}
        for this_res in results:
            res_dict = json.loads(this_res['payload'])
            assert res_dict['statusCode'] == 200
            for key in fields:
                requests[key].append(res_dict[key])
            preds[res_dict['learner']][:, res_dict['i_rep']] = res_dict['preds']
    requests = pd.DataFrame(requests)
    return preds, requests


def _extract_lambda_metrics(results):
    df = pd.DataFrame()
    for idx, this_res in enumerate(results):
        logs_list = base64.b64decode(this_res['log']).decode('utf-8').split('\n')
        report_line = logs_list[-2]
        assert report_line.startswith('REPORT RequestId')
        report_dict = dict(x.split(': ') for x in filter(None, report_line.split('\t')))
        for key, value in report_dict.items():
            if key in ['Duration', 'Billed Duration', 'Init Duration']:
                xx = value.split(' ')
                assert xx[1] == 'ms'
                report_dict[key] = float(xx[0])
            elif key in ['Memory Size', 'Max Memory Used']:
                xx = value.split(' ')
                assert xx[1] == 'MB'
                report_dict[key] = float(xx[0])
            else:
                assert key == 'REPORT RequestId'
        this_df = pd.DataFrame.from_dict(report_dict, orient='index').transpose()
        this_df.rename(columns={'REPORT RequestId': 'RequestId'}, inplace=True)
        df = df.append(this_df)
    df['Billed Duration GBSeconds'] = df['Billed Duration'] / 1000 * df['Memory Size'] / 1000

    return df
