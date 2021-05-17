import xgboost as xgb
import numpy as np
from Ques_2 import get_features, initialize_w2v_model, return_text, get_mean_vector, get_dicts
from bm25_performance import get_avg_ndcg
from retrieval_models import read_tsv, txt_write
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt


def get_train_test_data(lines_list):
    X_passage = []
    y_target = []

    relevancy_list = []
    query_qid = dict()
    passage_pid = dict()
    NUM = 0
    for line_item in lines_list:
        if (NUM < 2000000):

            qid, pid, query, passage, relevancy = int(line_item[0]), line_item[1], line_item[2], line_item[3], float(
                line_item[4])
            X_passage.append([query, passage])
            y_target.append(relevancy)

            relevancy_list.append(relevancy)
            NUM += 1
            if (query not in query_qid):
                query_qid[query] = qid
            if (passage not in passage_pid):
                passage_pid[passage] = pid

        else:
            break
    return X_passage, y_target, relevancy_list, passage_pid, query_qid


def get_features(X_passage, w2v_model):
    X_train = []

    for x in X_passage:
        query, passage = return_text(x[0]), return_text(x[1])

        query_vector = get_mean_vector(w2v_model, query)
        passage_vector = get_mean_vector(w2v_model, passage)

        product = np.dot(query_vector, passage_vector)

        query_norm = np.linalg.norm(query_vector)
        pass_norm = np.linalg.norm(passage_vector)

        dist = np.sqrt(np.sum(np.square(query_vector - passage_vector)))
        norm_plus = query_norm * pass_norm
        if (norm_plus == 0):
            score = 0
        else:
            score = product / norm_plus

        manhhaton = np.sum(np.abs(query_norm - pass_norm))

        X_train.append([score, dist, manhhaton])

    return X_train


def initialize_info(path):
    data = read_tsv(path)[1:]

    X_passage, y_target, relevancy_list, passage_pid, query_qid = get_train_test_data(data)
    return X_passage, y_target, relevancy_list, passage_pid, query_qid


def get_qid_pid_relevancy(qid_list, pid_list, relevancy_list):
    qid_pid_relevancy = dict()
    for i in range(len(qid_list)):
        qid = qid_list[i]
        pid = pid_list[i]
        relevancy = relevancy_list[i]

        if (qid not in qid_pid_relevancy):

            qid_pid_relevancy[qid] = {pid: relevancy}
        else:

            qid_pid_relevancy[qid].update({pid: relevancy})
    return qid_pid_relevancy


# X_passage 是[query,passage] 的list
def get_qid_candidates_target(X_passage, y_target, query_qid, passage_pid, trained):
    if (trained):
        sm = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
        X_passage, y_target = sm.fit_resample(X_passage, y_target)

    qid_candidates_target = dict()
    # with open(path, 'a', encoding='utf-8') as f:
    for i in range(len(X_passage)):
        query, passage = X_passage[i][0], X_passage[i][1]
        target = y_target[i]

        qid = query_qid[query]
        pid = passage_pid[passage]
        if qid not in qid_candidates_target:
            qid_candidates_target[qid] = {pid: target}
        else:
            qid_candidates_target[qid].update({pid: target})
    return qid_candidates_target


def get_txt_for_model(path, name, trained):
    print('111')
    X_passage, y_target, relevancy_list, passage_pid, query_qid = initialize_info(path)
    print('222')
    qid_candidates_target = get_qid_candidates_target(X_passage, y_target, query_qid, passage_pid, trained)
    qid_candidates_target = sorted(qid_candidates_target.items(), key=lambda x: x[0])
    print(len(qid_candidates_target))
    qid_NUM = len(qid_candidates_target)

    # [(2962, {'1588001': 0.0, '7430004': 0.0, '6412704': 0.0, '658625': 1.0, '7314664': 1.0}), (4696, {'5103343': 0.0, '7716223': 1.0}), (8701, {'758797': 0.0, '4163143': 0.0, '7990420': 1.0}), (10264, {'1817271': 0.0, '7791854': 1.0}), (12903, {'8038755': 0.0, '4230500': 0.0, '6381284': 1.0}), (14947, {'2250780': 0.0, '1356828': 0.0, '7541530': 1.0})
    path = name + '.txt'
    pid_passage = {v: k for k, v in passage_pid.items()}
    qid_query = {v: k for k, v in query_qid.items()}
    # 所有的query id已经按照从小到大排序了，也经过了0-1的平衡筛选
    # 回复了一个query-对多个passage id与其相关性01
    w2v_model = initialize_w2v_model()
    print(qid_NUM)

    pid_list = []
    qid_list = []
    rele_list = []
    with open(path, 'a', encoding='utf-8') as f:
        for i in range(qid_NUM):

            item = qid_candidates_target[i]
            qid = item[0]
            query = qid_query[qid]

            candidates_target = item[1]
            candidates_pid_list = list(candidates_target.keys())  # list 是所有 candidates 的passage pid

            for pid in candidates_pid_list:
                passage = pid_passage[pid]
                X_passage = [[query, passage]]
                target = candidates_target[pid]

                feature_vector = get_features(X_passage, w2v_model)[0]

                fea_0 = str(feature_vector[0])
                fea_1 = str(feature_vector[1])
                fea_2 = str(feature_vector[2])

                text = str(int(target)) + ' ' + 'qid:' + str(qid) + ' ' \
                       + '1:' + fea_0 + ' ' \
                       + '2:' + fea_1 + ' 3:' + fea_2 + ' \n'

                f.write(text)
                pid_list.append(pid)
                qid_list.append(qid)
                rele_list.append(target)
        return qid_candidates_target, pid_list, qid_list, rele_list


def train_model():
    train_data = xgb.DMatrix('train.txt')

    # param = {'max_depth': 10, 'eta': 0.2, 'verbosity': 2, 'objective': 'rank:map', 'num_round': 1e6}
    # mean
    # for average precision is:
    #     [0.11186086223107054, 0.11186086223107054, 0.11186086223107054, 0.11186086223107054, 0.11186086223107054]
    # mean
    # for ndcg is:
    #     [0.8040197953819346, 0.8040197953819346, 0.8040197953819346, 0.8040197953819346, 0.8040197953819346]


    param = {'max_depth': 2, 'eta': 0.2, 'verbosity': 2, 'objective': 'rank:map', 'num_round': 1e6,'gamma':0,'subsample':1}
    #mean for average precision is :
    # [0.12037364597406822, 0.12037364597406822, 0.12037364597406822, 0.12037364597406822, 0.12037364597406822]
    # mean for ndcg is :
    # [0.846373406433203, 0.846373406433203, 0.846373406433203, 0.846373406433203, 0.846373406433203]
    model = xgb.train(param, train_data)

    ax = xgb.plot_importance(model, color='red')
    fig = ax.figure
    fig.set_size_inches(20, 20)
    plt.show()
    return model


if __name__ == '__main__':
    train_path = 'dataset/train_data.tsv'
    test_path = 'dataset/validation_data.tsv'

    get_txt_for_model(test_path, trained=True, name='train')
    qid_candidates_target, pid_list, qid_list, relevancy_list = get_txt_for_model(test_path, trained=False, name='test')
    # train model
    model = train_model()


    test_data = xgb.DMatrix('test.txt')

    preds_list = model.predict(test_data)
    qid_pid_score, qid_pid_relevancy = get_dicts(qid_list, pid_list, preds_list, relevancy_list)

    qid_keys = list(qid_pid_score.keys())
    ave_sum_list = [0, 0, 0, 0, 0]
    ndcg_sum_list = [0, 0, 0, 0, 0]
    for qid in qid_keys:

        pid_score = qid_pid_score[qid]
        pid_relevancy = qid_pid_relevancy[qid]

        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        pid_relevancy = sorted(pid_relevancy.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        pid_lm = []
        pid_pref = []
        for i in range(len(pid_score)):
            pid_lm.append(pid_score[i][0])
            pid_pref.append(pid_relevancy[i][0])

        txt_write(qid=qid, pid_score=pid_score, function_name='LM')
        k_list = [20,40,60,80,100]
        for i in range (len(k_list)):
            k = k_list[i]
            ave_pre, ndcg = get_avg_ndcg(qid, pid_lm, pid_pref, qid_pid_relevancy, k)
            # print('qid = ', qid, 'k = ', k, ' average precision = ', ave_pre, ' ndgc = ', ndcg)
            ave_sum_list[i] += ave_pre
            ndcg_sum_list[i] += ndcg

    N = len(qid_keys)
    mean_pre = [x/N for x in ave_sum_list]
    mean_ndcg = [x/N for x in ndcg_sum_list]
    print('mean for average precision is :')
    print(mean_pre)
    print('mean for ndcg is :')
    print(mean_ndcg)
