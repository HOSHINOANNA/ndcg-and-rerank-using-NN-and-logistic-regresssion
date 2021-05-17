from retrieval_models import read_tsv
from logistic_regression import Logistic_Regression, Logistic_Regression_Predict
from imblearn.under_sampling import RandomUnderSampler
from nltk.tokenize import word_tokenize
from language_model import txt_write
from gensim.models.word2vec import Word2Vec
from bm25_performance import get_avg_ndcg

import re
import numpy as np
import gensim


def get_train_test_data(lines_list):
    X_passage = []
    y_target = []
    pid_list = []
    qid_list = []
    relevancy_list = []
    NUM = 0
    for line_item in lines_list:
        if(NUM<2000000):

            qid, pid, query, passage, relevancy = line_item[0], line_item[1], line_item[2], line_item[3], float(
                line_item[4])
            X_passage.append([query, passage])
            y_target.append(relevancy)
            pid_list.append(pid)
            qid_list.append(qid)
            relevancy_list.append(relevancy)
            NUM+=1
        else: break
    return X_passage, y_target, pid_list, qid_list, relevancy_list


def return_text(passage):
    line = re.sub("[\s+\.\!\/_,$|%^*(+\"\'><):-]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  " ", passage).lower()
    words = word_tokenize(line)

    return words


def initialize_info(path):
    data = read_tsv(path)[1:]

    X_passage, y_target, pid_list, qid_list, relevancy_list = get_train_test_data(data)
    return X_passage, y_target, pid_list, qid_list, relevancy_list


def reformat_data(X_passage, y_target, train):
    if (train):
        sm = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
        X_passage, y_target = sm.fit_resample(X_passage, y_target)
    y = np.array(y_target)
    m = len(y)
    X = np.array(get_features(X_passage))
    X = np.c_[np.ones((m, 1)), X]
    return X, y


def initialize_w2v_model():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model


def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.key_to_index]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros(300)


def get_features(X_passage):
    X_train = []
    w2v_model = initialize_w2v_model()
    print('CC')
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

        manhhaton = np.sum(np.abs(query_vector - passage_vector))

        X_train.append([score, dist, manhhaton])


    return X_train


def train_model(X, y):

    theta_0 = [0, 0, 0, 0]  # theta初始化
    epochs = 3e3  # 迭代次数
    print(theta_0)

    alpha = 0.001
    model_lr, decent_cost_function = Logistic_Regression(X, y, alpha, theta_0, epochs)

    print('alpha = 0.001 :', decent_cost_function)

    alpha = 0.1
    model_lr, decent_cost_function = Logistic_Regression(X, y, alpha, theta_0, epochs)
    print('alpha = 0.1 :', decent_cost_function)

    alpha = 0.01
    model_lr, decent_cost_function = Logistic_Regression(X, y, alpha, theta_0, epochs)

    print('alpha = 0.01 :', decent_cost_function)
    return model_lr


def get_dicts(qid_list, pid_list, y_prop, relevancy_list):
    qid_pid_prop = dict()
    qid_pid_relevancy = dict()
    for i in range(len(qid_list)):
        qid = qid_list[i]
        pid = pid_list[i]
        property = y_prop[i]
        relevancy = relevancy_list[i]
        if (qid not in qid_pid_prop):
            qid_pid_prop[qid] = {pid: property}
            qid_pid_relevancy[qid] = {pid: relevancy}
        else:
            qid_pid_prop[qid].update({pid: property})
            qid_pid_relevancy[qid].update({pid: relevancy})
    return qid_pid_prop, qid_pid_relevancy


if __name__ == '__main__':
    train_path = 'dataset/train_data.tsv'
    # train_path = 'dataset/validation_data.tsv'
    #4300000
    #1103039

    X_passage, y_target, __, __, __ = initialize_info(train_path)
    print(len(X_passage))
    print(X_passage[:10])
    print(y_target[:10])
    X, y = reformat_data(X_passage, y_target, train=True)

    print('222')
    model_lr = train_model(X, y)
    print('333')

    test_path = 'dataset/validation_data.tsv'
    X_passage, y_target, pid_list, qid_list, relevancy_list = initialize_info(test_path)
    print('444')
    X, y = reformat_data(X_passage, y_target, train=False)
    print('555')

    y_prop = Logistic_Regression_Predict(model_lr, X)

    qid_pid_probability, qid_pid_relevancy = get_dicts(qid_list, pid_list, y_prop, relevancy_list)
    print(len(qid_list))
    j = 0
    #qid_list 的长度1103039 是按照顺序的长度
    # 这里应该用字典的key
    #这行代码后只能用字典
    qid_list = list(qid_pid_probability.keys())
    print(len(qid_list))
    ave_sum_list = [0, 0, 0, 0, 0]
    ndcg_sum_list = [0, 0, 0, 0, 0]
    for qid in qid_list:

        pid_probability = qid_pid_probability[qid]
        pid_relevancy = qid_pid_relevancy[qid]

        pid_probability = sorted(pid_probability.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        pid_relevancy = sorted(pid_relevancy.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        pid_lr = []
        pid_pref = []
        for i in range(len(pid_probability)):
            pid_lr.append(pid_probability[i][0])
            pid_pref.append(pid_relevancy[i][0])


        txt_write(qid=qid, pid_score=pid_probability, function_name='LR')
        k_list = [20,40,60,80,100]
        for i in range(len(k_list)):
            k = k_list[i]
            ave_pre, ndcg = get_avg_ndcg(qid, pid_lr, pid_pref, qid_pid_relevancy, k)
            ave_sum_list[i] += ave_pre
            ndcg_sum_list[i] += ndcg

    N = len(qid_list)
    mean_pre = [x / N for x in ave_sum_list]
    mean_ndcg = [x / N for x in ndcg_sum_list]
    print('mean for average precision is :')
    print(mean_pre)
    print('mean for ndcg is :')
    print(mean_ndcg)

