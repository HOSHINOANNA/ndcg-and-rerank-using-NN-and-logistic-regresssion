import tensorflow as tf

from Ques_2 import initialize_info, reformat_data, get_dicts
from bm25_performance import get_avg_ndcg
from retrieval_models import txt_write

if __name__ == '__main__':

    train_path = 'dataset/train_data.tsv'
    # train_path = 'dataset/validation_data.tsv'

    # 4300000
    # 1103039

    X_passage, y_target, __, __, __ = initialize_info(train_path)
    print(len(X_passage))
    print(X_passage[:10])
    print(y_target[:10])
    X_train, y_train = reformat_data(X_passage, y_target, train=True)

    print('aaa')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='sgd', loss='mse')
    # This builds the model for the first time:

    model.fit(X_train, y_train, batch_size=1000, epochs=100)

    # test_


    test_path = 'dataset/validation_data.tsv'
    X_passage, y_target,pid_list, qid_list, relevancy_list = initialize_info(test_path)
    X_test, y_test = reformat_data(X_passage, y_target, train=False)
    y_predict = model.predict(X_test)
    y_predict = [b for a in y_predict for b in a]
    qid_pid_score, qid_pid_relevancy = get_dicts(qid_list, pid_list, y_predict, relevancy_list)
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

        txt_write(qid=qid, pid_score=pid_score, function_name='NN')
        k_list = [20,40,60,80,100]
        for i in range(len(k_list)):
            k = k_list[i]
            ave_pre, ndcg = get_avg_ndcg(qid, pid_lm, pid_pref, qid_pid_relevancy, k)
            # print('qid = ', qid, 'k = ', k, ' average precision = ', ave_pre, ' ndgc = ', ndcg)
            ave_sum_list[i] += ave_pre
            ndcg_sum_list[i] += ndcg

    N = len(qid_keys)
    mean_pre = [x / N for x in ave_sum_list]
    mean_ndcg = [x / N for x in ndcg_sum_list]
    print('mean for average precision is :')
    print(mean_pre)
    print('mean for ndcg is :')
    print(mean_ndcg)

    
