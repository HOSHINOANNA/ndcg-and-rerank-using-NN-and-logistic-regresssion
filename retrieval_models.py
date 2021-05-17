import csv
import math
import re

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

PASSAGE_NUM = 0
stop_words = set(stopwords.words('english'))


def read_tsv(path):
    tsv_line = []
    csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)

    with open(path) as csvfile:
        file_list = csv.reader(csvfile, 'mydialect')
        for line in file_list:
            tsv_line.append(line)
    return tsv_line


def get_inverted_index(pid_passage, candidates_id):
    inverted_index = {}

    englishStemmer = SnowballStemmer("english")

    for id in candidates_id:
        passage = pid_passage[id]
        text = re.sub("[\s+\.\!\/_,$|%^*(+\"\'><):-]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                      " ", passage).lower()
        words = word_tokenize(text)
        position = 0
        for w in words:
            position = position + 1
            w = englishStemmer.stem(w)
            if w not in stop_words:
                # index doesnt save this word
                if w not in inverted_index:
                    inverted_index[w] = {id: 1}
                # index has this word
                else:
                    # index appears in new document
                    if id not in inverted_index[w]:
                        dict1 = inverted_index[w]
                        dict1.update({id: 1})
                        inverted_index[w] = dict1
                    else:
                        # print('a;',inverted_index[w])
                        # print('b',inverted_index[w][id])
                        inverted_index[w][id] += 1

                        # inverted_index[w] = inverted_index[w]+[id]
    return inverted_index


def get_passage_dictionary(lines_list):
    qid_candidate_p = dict()
    pid_passage = dict()
    for line_item in lines_list:

        if line_item[1] not in pid_passage:
            pid_passage[line_item[1]] = line_item[3]
        if line_item[0] not in qid_candidate_p:
            qid_candidate_p[line_item[0]] = [line_item[1]]
        else:
            qid_candidate_p[line_item[0]] = qid_candidate_p[line_item[0]] + [line_item[1]]

    return pid_passage, qid_candidate_p


def get_query_dictionary(lines_list):
    qid_passage = dict()
    for line_item in lines_list:
        qid_passage[line_item[0]] = line_item[1]
    return qid_passage


def get_passage_words(passage):
    words = passage.split()
    return len(words)


def reformat_query(query):
    # text-preprocessing
    keyword_frequency = dict()
    # remove punctuation and others
    text = re.sub("[\s+\.\!\/_,$|%^*(+\"\'><):-]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  " ", query).lower()
    words_query = word_tokenize(text)
    englishStemmer = SnowballStemmer("english")
    for i in range(len(words_query)):
        w = words_query[i]
        if w not in stop_words:
            w = englishStemmer.stem(w)
            if w not in keyword_frequency:
                keyword_frequency[w] = 1
            else:
                keyword_frequency[w] = keyword_frequency[w] + 1

    return keyword_frequency


def vector_space_score(keywords_freq, inverted_index, candidates_id):
    pid_score = dict()
    keyword_list = list(keywords_freq.keys())
    N = len(candidates_id) + 1
    for keyword in keyword_list:
        if keyword in inverted_index:
            pass
        # add keywords to inverted index
        else:
            inverted_index[keyword] = {0, 1}

    all_words_len = len(inverted_index)
    for p_id in candidates_id:

        query_vector = np.zeros(all_words_len)
        passage_vector = np.zeros(all_words_len)
        i = 0
        for word in inverted_index:
            if p_id in inverted_index[word]:
                tf = inverted_index[word][p_id]
                idf = math.log(N / len(inverted_index[word]))
                passage_vector[i] = tf * idf
            if word in keyword_list:
                # print('aaa')
                tf = keywords_freq[word]
                # print(len(inverted_index[word]))
                idf = math.log(N / len(inverted_index[word]))
                # print(idf)
                query_vector[i] = tf * idf
            i = i + 1

        product = np.dot(query_vector, passage_vector)

        query_norm = np.linalg.norm(query_vector)
        pass_norm = np.linalg.norm(passage_vector)
        score = product / (query_norm * pass_norm)
        pid_score[p_id] = score
    return pid_score


def get_pid_vector(inverted_index):
    pid_all_freq = dict()
    for k in inverted_index:
        for pid in inverted_index[k]:
            if pid not in pid_all_freq:
                pid_all_freq[pid] = [inverted_index[k][pid]]
            else:
                pid_all_freq[pid] = pid_all_freq[pid] + [inverted_index[k][pid]]
    return pid_all_freq


def BM25_score(keywords_freq, inverted_index, candidates_id, pid_passage):
    pid_score = dict()

    N = len(candidates_id)
    k1 = 1.2
    b = 0.75
    k2 = 100
    avgdl = get_avgdl(candidates_id, pid_passage)

    for pid in candidates_id:
        dl = getdl(pid, pid_passage)
        K = k1 * (1 - b + b * (dl / avgdl))
        score = 0
        for keyword in keywords_freq:
            if keyword not in inverted_index:
                score += 0
            else:
                ni = len(inverted_index[keyword])
                fi = inverted_index[keyword][pid] if pid in inverted_index[keyword] else 0
                qfi = keywords_freq[keyword]

                idf = math.log((N - ni + 0.5) / (ni + 0.5))
                w1 = ((k1 + 1) * fi) / (K + fi)
                w2 = ((k2 + 1) * qfi) / (k2 + qfi)
                score += idf * w1 * w2

        pid_score[pid] = score
    return pid_score


def txt_write(qid, pid_score, function_name):
    path = function_name + '.txt'
    with open(path, 'a', encoding='utf-8') as f:
        loop = 100 if len(pid_score) > 100 else len(pid_score)
        for i in range(loop):
            pid = pid_score[i][0]
            score = str(pid_score[i][1])
            text = str(qid) + ' ' + 'A1' + ' ' + str(pid) + ' ' + 'rank' + str(i + 1) + ' ' + str(score) + " " + function_name + '\n'
            f.write(text)


def get_avgdl(candidates_id, pid_passage):
    passage_num = len(candidates_id)
    sum = 0
    for pid in candidates_id:
        passage = pid_passage[pid]
        a = passage.split()
        sum += len(a)
    return sum / passage_num


def getdl(pid, pid_passage):
    passage = pid_passage[pid]
    a = passage.split()
    return len(a)


def initialize():
    tsv_path = 'dataset/candidate_passages_top1000.tsv'
    tsv_line = read_tsv(tsv_path)
    # pid_passage <dict>,qid_candidates_pi<dict>
    pid_passage, qid_candidates_pid = get_passage_dictionary(tsv_line)
    query_path = 'dataset/test-queries.tsv'
    query_line = read_tsv(query_path)
    qid_query = get_query_dictionary(query_line)
    return pid_passage, qid_candidates_pid, qid_query


def run_VS():
    pid_passage, qid_candidates_pid, qid_query = initialize()

    for qid in qid_query:
        # qid = '903469'
        candidates_id = qid_candidates_pid[qid]

        inverted_index = get_inverted_index(pid_passage, candidates_id)
        # Every keyword with term frequency
        keywords_freq = reformat_query(qid_query[qid])

        query = qid_query[qid]
        pid_score = vector_space_score(keywords_freq, inverted_index, candidates_id)
        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        txt_write(qid=qid, pid_score=pid_score, function_name='VS')


def run_bm25():
    pid_passage, qid_candidates_pid, qid_query = initialize()

    for qid in qid_query:
        candidates_id = qid_candidates_pid[qid]

        inverted_index = get_inverted_index(pid_passage, candidates_id)
        keywords_freq = reformat_query(qid_query[qid])

        pid_score = BM25_score(keywords_freq, inverted_index, candidates_id, pid_passage)
        pid_score = sorted(pid_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        txt_write(qid=qid, pid_score=pid_score, function_name='BM25')


if __name__ == '__main__':
    run_VS()
    run_bm25()
