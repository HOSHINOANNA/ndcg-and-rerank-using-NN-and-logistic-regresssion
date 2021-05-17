from retrieval_models import read_tsv,get_inverted_index, reformat_query, BM25_score
import numpy as np


def get_passage_dictionary(lines_list):
    qid_candidate_relevancy = dict()

    pid_passage = dict()
    qid_query = dict()
    for line_item in lines_list:
        qid, pid, query, passage, relevancy = line_item[0], line_item[1], line_item[2], line_item[3], float(line_item[4])
        if qid not in qid_query:
            qid_query[qid] = query
        # passage id
        if pid not in pid_passage:
            pid_passage[pid] = passage
        # query id
        if qid not in qid_candidate_relevancy:
            qid_candidate_relevancy[qid] = {pid: relevancy}
        else:
            qid_candidate_relevancy[qid].update({pid: relevancy})

    return pid_passage, qid_query, qid_candidate_relevancy


def get_average_precision(qid, pid_list, qid_candidate_relevancy,k):
    candidate_relevancy = qid_candidate_relevancy[qid]
    relevant_documents_number = 0
    average_precision = 0
    sum_precision = []

    # try:
    #     one_list = list(candidate_relevancy.keys())[list(candidate_relevancy.values()).index(1)]
    #     relevant_documents_number = len(one_list)
    # except ValueError:
    #     relevant_documents_number = 0

    temp_relevance_num = 0
    # if (relevant_documents_number == 0):
    #     average_precision = 0
    #     return average_precision
    # else:
    loop = k if len(pid_list) > k else len(pid_list)

    for i in range(loop):
        pid = pid_list[i]
        relevancy = qid_candidate_relevancy[qid][pid]
        if relevancy == 1:
            temp_relevance_num += 1
            precision = temp_relevance_num / (i + 1)
            sum_precision.append(precision)

    if(len(sum_precision)<1):
        average_precision = 0
    else:
        average_precision = np.mean(sum_precision)

    return average_precision


def get_dcg(qid, pid_list, qid_candidate_relevancy,k):
    loop = k if len(pid_list) > k else len(pid_list)

    sum_dcg = 0
    for i in range(loop):
        pid = pid_list[i]
        relevancy = qid_candidate_relevancy[qid][pid]
        gain = 2 ** (relevancy) - 1
        position = i + 1
        position_discount = np.log2(position + 1)
        dcg = gain/position_discount
        sum_dcg = sum_dcg + dcg
    return sum_dcg



def get_avg_ndcg(qid, pid_bm25,pid_pref, qid_candidate_relevancy,k):
    ave_pre = get_average_precision(qid, pid_bm25, qid_candidate_relevancy,k)

    dcg = get_dcg(qid, pid_bm25, qid_candidate_relevancy, k)
    idcg = get_dcg(qid, pid_pref, qid_candidate_relevancy, k)
    ndcg = dcg / idcg
    # print(ave_pre,dcg,idcg)
    return ave_pre,ndcg


def run():

    validation_path = 'dataset/validation_data.tsv'
    # train_data = read_tsv(train_path)[1:]
    validation_data = read_tsv(validation_path)[1:]
    pid_passage, qid_query, qid_candidate_relevancy = get_passage_dictionary(validation_data)

    ave_sum_list = [0, 0, 0, 0, 0]
    ndcg_sum_list = [0, 0, 0, 0, 0]
    print(len(qid_query))
    j = 0
    for qid in qid_query:
        j+=1
        print(j)

        candidates_id = qid_candidate_relevancy[qid].keys()

        inverted_index = get_inverted_index(pid_passage, candidates_id)
        keywords_freq = reformat_query(qid_query[qid])

        pid_score = BM25_score(keywords_freq, inverted_index, candidates_id, pid_passage)
        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        candidate_relevancy = qid_candidate_relevancy[qid]
        candidate_relevancy = sorted(candidate_relevancy.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        #bm25的分数从高到低
        pid_bm25 = []
        #相关度0-1 从高到低 只要id！！ 不要分
        pid_pref = []
        for i in range (len(pid_score)):
            pid_bm25.append(pid_score[i][0])
            pid_pref.append(candidate_relevancy[i][0])

        k_list = [20,40,60,80,100]

        for i in range(len(k_list)):
            k = k_list[i]
            ave_pre,ndcg = get_avg_ndcg(qid, pid_bm25,pid_pref, qid_candidate_relevancy,k)
            # print('qid = ',qid,'k = ',k,' average precision = ',ave_pre,' ndgc = ',ndcg)
            ave_sum_list[i]+=ave_pre
            ndcg_sum_list[i]+=ndcg

    N = len(qid_query)
    mean_pre = [x / N for x in ave_sum_list]
    mean_ndcg = [x / N for x in ndcg_sum_list]
    print('mean for average precision is :')
    print(mean_pre)
    print('mean for ndcg is :')
    print(mean_ndcg)


if __name__ == '__main__':
    run()
