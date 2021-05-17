from retrieval_models import initialize, get_inverted_index, reformat_query, txt_write, getdl
import numpy as np
import math


def score_Dirichlet(keywords_freq, inverted_index, candidates_id, pid_passage):
    pid_score = dict()
    keyword_list = list(keywords_freq.keys())
    miu = 2000
    # inverted index
    for keyword in keyword_list:
        if keyword in inverted_index:
            dict1 = inverted_index[keyword]
            dict1.update({'001': keywords_freq[keyword]})
            inverted_index[keyword] = dict1

        else:
            inverted_index[keyword] = {'001': keywords_freq[keyword]}

        C = 0
    for word in inverted_index:
        C += sum(inverted_index[word].values())
    # print(C)
    # print('len',len(inverted_index))

    for pid in candidates_id:
        D = getdl(pid, pid_passage)
        score = 0
        lamda = D / (D + miu)
        for keyword in keyword_list:
            cf = sum(inverted_index[keyword].values())
            fq = inverted_index[keyword][pid] if pid in inverted_index[keyword] else 0

            score = score + math.log(lamda * (fq / D) + (1 - lamda) * (cf / C))
        pid_score[pid] = score

    return pid_score


def score_Laplace(keywords_freq, inverted_index, candidates_id, pid_passage):
    pid_score = dict()
    keyword_list = list(keywords_freq.keys())

    for keyword in keyword_list:
        if keyword in inverted_index:
            dict1 = inverted_index[keyword]
            dict1.update({'001': keywords_freq[keyword]})
            inverted_index[keyword] = dict1

        else:
            inverted_index[keyword] = {'001': keywords_freq[keyword]}

    V = len(inverted_index)

    for pid in candidates_id:
        D = getdl(pid, pid_passage)

        score = 0
        for keyword in keyword_list:
            fq = inverted_index[keyword][pid] if pid in inverted_index[keyword] else 0
            score = score + math.log((fq + 1) / (D + V))
        pid_score[pid] = score

    return pid_score


def score_Lindstone(keywords_freq, inverted_index, candidates_id, pid_passage):
    pid_score = dict()
    keyword_list = list(keywords_freq.keys())
    e = 0.5

    for keyword in keyword_list:
        if keyword in inverted_index:
            dict1 = inverted_index[keyword]
            dict1.update({'001': keywords_freq[keyword]})
            inverted_index[keyword] = dict1

        else:
            inverted_index[keyword] = {'001': keywords_freq[keyword]}

    V = len(inverted_index)

    for pid in candidates_id:
        D = getdl(pid, pid_passage)

        score = 0
        for keyword in keyword_list:
            fq = inverted_index[keyword][pid] if pid in inverted_index[keyword] else 0
            score = score + math.log((fq + e) / (D + e * V))
        pid_score[pid] = score

    return pid_score


def run_Dirichlet():
    pid_passage, qid_candidates_pid, qid_query = initialize()

    for qid in qid_query:
        candidates_id = qid_candidates_pid[qid]

        inverted_index = get_inverted_index(pid_passage, candidates_id)
        keywords_freq = reformat_query(qid_query[qid])

        pid_score = score_Dirichlet(keywords_freq, inverted_index, candidates_id, pid_passage)
        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        txt_write(qid=qid, pid_score=pid_score, function_name='LM-Dirichlet')


def run_Laplace():
    pid_passage, qid_candidates_pid, qid_query = initialize()

    for qid in qid_query:
        candidates_id = qid_candidates_pid[qid]

        inverted_index = get_inverted_index(pid_passage, candidates_id)
        keywords_freq = reformat_query(qid_query[qid])

        pid_score = score_Laplace(keywords_freq, inverted_index, candidates_id, pid_passage)
        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        txt_write(qid=qid, pid_score=pid_score, function_name='LM-Laplace')


def run_Lindstone():
    pid_passage, qid_candidates_pid, qid_query = initialize()

    for qid in qid_query:
        candidates_id = qid_candidates_pid[qid]

        inverted_index = get_inverted_index(pid_passage, candidates_id)
        keywords_freq = reformat_query(qid_query[qid])

        pid_score = score_Lindstone(keywords_freq, inverted_index, candidates_id, pid_passage)
        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        txt_write(qid=qid, pid_score=pid_score, function_name='LM-Lindstone')


if __name__ == '__main__':
    # use this to run
    run_Dirichlet()
    run_Laplace()
    run_Lindstone()
