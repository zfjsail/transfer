from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
import os
from os.path import join
from collections import defaultdict as dd
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text
from keras.preprocessing.sequence import pad_sequences

from utils import feature_utils
from utils import data_utils
from utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class CNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle, args, use_emb=True):

        self.file_dir = file_dir

        self.matrix_title_size = matrix_size1
        self.matrix_author_size = matrix_size2

        # load training pairs

        pos_pairs = data_utils.load_json(file_dir, 'pos_person_pairs.json')
        neg_pairs = data_utils.load_json(file_dir, 'neg_person_pairs.json')
        pairs = pos_pairs + neg_pairs
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        self.person_dict = data_utils.load_json(file_dir, "ego_person_dict.json")

        self.use_emb = use_emb
        if self.use_emb:
            self.pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))
        self.tokenizer = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        X_long = []
        X_short = []
        nn_pos = 0
        nn_neg = 0
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d %d %d', i, nn_pos, nn_neg)
            # cpaper, npaper = pair
            aid, mid = pair['aid'], pair['mid']
            aperson = self.person_dict.get(aid, {})
            mperson = self.person_dict.get(mid, {})
            # matrix1, nn1 = self.org_to_matrix(aperson.get('org', ''), mperson.get('org', ''), matrix_size1)
            matrix1, nn1 = self.paper_to_matrix(aperson.get('pubs', []), mperson.get('pubs', []), matrix_size1)
            # print("matrix1", matrix1)

            matrix1 = feature_utils.scale_matrix(matrix1)
            X_long.append(matrix1)
            matrix2, nn2 = self.venue_to_matrix(aperson.get('venue', ''), mperson.get('venue', ''), matrix_size2)
            # print("matrix2", matrix2)
            matrix2 = feature_utils.scale_matrix(matrix2)
            X_short.append(matrix2)
            # if y[i][0] == 1:
            #     nn_pos += (nn1 + nn2)
            # else:
            #     nn_neg += (nn1 + nn2)

        self.X_long = X_long
        self.X_short = X_short
        self.Y = labels

        print("shuffle", shuffle)
        if shuffle:
            self.X_long, self.X_short, self.Y = sklearn.utils.shuffle(
                self.X_long, self.X_short, self.Y,
                random_state=seed
            )

        self.N = len(self.Y)

        n_train = args.train_num
        n_test = args.test_num

        train_data = {}
        train_data["x1"] = self.X_long[:n_train]
        train_data["x2"] = self.X_short[:n_train]
        train_data["y"] = self.Y[:n_train]
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1"] = self.X_long[n_train:(n_train+n_test)]
        test_data["x2"] = self.X_short[n_train:(n_train+n_test)]
        test_data["y"] = self.Y[n_train:(n_train+n_test)]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1"] = self.X_long[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2"] = self.X_short[n_train+n_test:(n_train+n_test*2)]
        valid_data["y"] = self.Y[n_train+n_test:(n_train+n_test*2)]
        print("valid labels", len(valid_data["y"]))

        print("train positive samples", sum(train_data["y"]))
        print("test positive samples", sum(test_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "author_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "author_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "author_valid.pkl")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_long[idx], self.X_short[idx], self.Y[idx]

    def org_to_matrix(self, title1, title2, max_size):
        twords1 = feature_utils.get_words(title1, remove_stopwords=True)[: max_size]
        twords2 = feature_utils.get_words(title2, remove_stopwords=True)[: max_size]

        matrix = -np.ones((max_size, max_size))
        nn1 = 0
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                v = 1 if word1 == word2 else -1
                # if word1 == word2:
                #     matrix[i][j] = 1.
                #     continue
                # v = sim_utils.sim_ngrams(word1, word2)
                # v = 2 * v - 1
                if v == 1:
                    nn1 += 1
                matrix[i][j] = v
        # print(twords1, twords2, nn1)

        return matrix, nn1

    def venue_to_matrix(self, o1, o2, max_size):
        # twords1 = utils.extract_venue_features(o1)[:max_size].split()
        # twords2 = utils.extract_venue_features(o2)[:max_size].split()
        avenue = [v['id'] for v in o1]
        mvenue = [v['id'] for v in o2]
        if self.use_emb:
            avenue = self.tokenizer.texts_to_sequences([" ".join(avenue)])[0][:max_size]
            # print("avenue", avenue)
            mvenue = self.tokenizer.texts_to_sequences([" ".join(mvenue)])[0][:max_size]
        avenue = avenue[: max_size]
        mvenue = mvenue[: max_size]

        matrix = -np.ones((max_size, max_size))
        nn1 = 0
        for i, avid in enumerate(avenue):
            # v1_dec = avid + '-v'
            # emb1 = lc.get(v1_dec)
            for j, mvid in enumerate(mvenue):
                v = -1
                if avid == mvid:
                    v = 1
                    nn1 += 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[avid].reshape(1, -1),
                                      self.pretrain_emb[mvid].reshape(1, -1))[0][0]
                    # print("v", v)
                # elif emb1 is None:
                #     continue
                # else:
                #     v2_dec = mvid + '-v'
                #     emb2 = lc.get(v2_dec)
                #     if emb2 is not None:
                #         v = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
                #         # print('cos', v)
                matrix[i, j] = v
        print(nn1, avenue, mvenue)
        return matrix, None

    def get_noisy_papers_test(self):
        return data_utils.load_json_lines(self.file_dir, 'noisy-papers-test.dat')


    def authors_to_matrix(self, authors1, authors2):
        matrix = -np.ones((self.matrix_author_size, self.matrix_author_size))
        author_num = int(self.matrix_author_size/2)
        try:
            for i in range(author_num):
                row = 2 * i
                a1 = authors1[i].lower().split()
                first_name1 = a1[0][0]
                last_name1 = a1[-1][0]
                col = row
                a2 = authors2[i].lower().split()
                first_name2 = a2[0][0]
                last_name2 = a2[-1][0]
                matrix[row][col] = feature_utils.name_equal(first_name1, first_name2)
                matrix[row][col+1] = feature_utils.name_equal(first_name1, last_name2)
                matrix[row+1][col] = feature_utils.name_equal(last_name1, first_name2)
                matrix[row+1][col+1] = feature_utils.name_equal(last_name1, last_name2)
        except Exception as e:
            pass
        return matrix

    def paper_to_matrix(self, ap, mp, max_size):
        if self.use_emb:
            apubs = self.tokenizer.texts_to_sequences([" ".join(ap)])[0][:max_size]
            mpubs = self.tokenizer.texts_to_sequences([" ".join(mp)])[0][:max_size]
            # print("apubs", apubs)
        else:
            apubs = ap[:max_size]
            mpubs = mp[:max_size]
        matrix = -np.ones((max_size, max_size))
        for i, apid in enumerate(apubs):
            # p1_dec = apid + '-p'
            # emb1 = lc.get(p1_dec)
            for j, mpid in enumerate(mpubs):
                v = -1
                if apid == mpid:
                    v = 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[apid].reshape(1, -1),
                                      self.pretrain_emb[mpid].reshape(1, -1))[0][0]
                    # print("v", v)
                # elif emb1 is None:
                #     continue
                # else:
                #     p2_dec = mpid + '-p'
                #     emb2 = lc.get(p2_dec)
                #     if emb2 is not None:
                #         v = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
                #         # print('cos', v)
                matrix[i, j] = v
        return matrix, None

    def get_id2cpapers(self):
        cpapers_train = data_utils.load_json_lines(self.file_dir, 'clean-papers-train.dat')
        cpapers_test = data_utils.load_json_lines(self.file_dir, 'clean-papers-test.dat')
        cpapers = cpapers_train + cpapers_test
        id2paper = {}
        for paper in cpapers:
            paper['id'] = str(paper['id'])
            pid = paper['id']
            id2paper[pid] = paper
        # data_utils.dump_json(id2paper, self.file_dir, 'clean-id2paper.json')
        return id2paper

    def build_cpapers_inverted_index(self):
        logger.info('build inverted index for cpapers')
        cpapers_train = data_utils.load_json_lines(self.file_dir, 'clean-papers-train.dat')
        cpapers_test = data_utils.load_json_lines(self.file_dir, 'clean-papers-test.dat')
        papers = cpapers_train + cpapers_test
        word2ids = dd(list)
        for paper in papers:
            pid = str(paper['id'])
            title = paper['title']
            words = feature_utils.get_words(title.lower(), window=self.build_index_window)
            for word in words:
                word2ids[word].append(pid)
        for word in word2ids:
            word2ids[word] = list(set(word2ids[word]))
        # data_utils.dump_json(word2ids, self.file_dir, 'clean-papers-inverted-index.json')
        logger.info('building inverted index completed')
        return word2ids

    def get_candidates_by_inverted_index(self, npaper, word2ids):
        title = npaper['title'].lower()
        words = feature_utils.get_words(title, window=self.build_index_window)
        cids_to_freq = dd(int)
        for word in words:
            if word in word2ids:
                cur_cids = word2ids[word]
                for cid in cur_cids:
                    cids_to_freq[cid] += 1
        sorted_items = sorted(cids_to_freq.items(), key=lambda kv: kv[1], reverse=True)[:20]
        cand_cids = [item[0] for item in sorted_items]
        return cand_cids


class AuthorRNNMatchDataset(Dataset):

    def __init__(self, file_dir, max_seq1_len, max_seq2_len, shuffle, seed, args):

        self.max_seq1_len = max_seq1_len
        self.max_seq2_len = max_seq2_len

        # load training pairs

        pos_pairs = data_utils.load_json(file_dir, 'pos_person_pairs.json')
        neg_pairs = data_utils.load_json(file_dir, 'neg_person_pairs.json')
        pairs = pos_pairs + neg_pairs
        self.labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        self.person_dict = data_utils.load_json(file_dir, "ego_person_dict.json")

        corpus = []
        nn_pos = 0
        nn_neg = 0

        # for i, pair in enumerate(pairs):
        #     if i % 100 == 0:
        #         logger.info('pairs to matrices %d %d %d', i, nn_pos, nn_neg)
        #     # cpaper, npaper = pair
        #     aid, mid = pair['aid'], pair['mid']
        #     aperson = self.person_dict.get(aid, {})
        #     mperson = self.person_dict.get(mid, {})
        #     corpus.append(aperson.get("pubs", []))
        #     corpus.append(mperson.get("pubs", []))
        #
        #     corpus.append([item["id"] for item in aperson.get("venue", [])])
        #     corpus.append([item["id"] for item in mperson.get("venue", [])])
        #
        # t = Tokenizer(num_words=100000)
        # t.fit_on_texts(corpus)

        t = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        self.vocab_size = len(t.word_counts)
        print("vocab size", self.vocab_size)
        # print("tokenizer", t.word_index)

        self.mag = [self.person_dict.get(pair["mid"], {}).get("pubs", []) for pair in pairs]
        self.aminer = [self.person_dict.get(pair["aid"], {}).get("pubs", []) for pair in pairs]
        self.mag = t.texts_to_sequences(self.mag)
        # print("self mag", self.mag)

        self.aminer = t.texts_to_sequences(self.aminer)
        self.mag = pad_sequences(self.mag, maxlen=self.max_seq1_len)
        self.aminer = pad_sequences(self.aminer, maxlen=self.max_seq1_len)

        self.mag_keywords = []
        self.aminer_keywords = []
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d %d %d', i, nn_pos, nn_neg)
            # cpaper, npaper = pair
            aid, mid = pair['aid'], pair['mid']
            avenue = [item["id"] for item in self.person_dict.get(aid, {}).get("venue", [])]
            mvenue = [item["id"] for item in self.person_dict.get(mid, {}).get("venue", [])]
            self.mag_keywords.append(mvenue)
            self.aminer_keywords.append(avenue)


        # self.mag_keywords = [self.person_dict.get(pair["mid"], {}).get("venue", []) for pair in pairs]
        # self.mag_keywords = [venue["id"] for venue in self.mag_keywords]
        # self.aminer_keywords = [self.person_dict.get(pair["aid"], {}).get("venue", []) for pair in pairs]
        # self.aminer_keywords = [venue["id"] for venue in self.aminer_keywords]

        self.mag_keywords = t.texts_to_sequences(self.mag_keywords)
        self.aminer_keywords = t.texts_to_sequences(self.aminer_keywords)

        self.mag_keywords = pad_sequences(self.mag_keywords, maxlen=max_seq2_len)
        self.aminer_keywords = pad_sequences(self.aminer_keywords, maxlen=max_seq2_len)

        if shuffle:
            self.mag, self.aminer, self.mag_keywords, self.aminer_keywords, self.labels = sklearn.utils.shuffle(
                self.mag, self.aminer, self.mag_keywords, self.aminer_keywords, self.labels,
                random_state=seed
            )

        self.N = len(self.labels)

        n_train = args.train_num
        n_test = args.test_num

        train_data = {}
        train_data["x1_seq1"] = self.mag[:n_train]
        train_data["x1_seq2"] = self.mag_keywords[:n_train]
        train_data["x2_seq1"] = self.aminer[:n_train]
        train_data["x2_seq2"] = self.aminer_keywords[:n_train]
        train_data["y"] = self.labels[:n_train]
        train_data["vocab_size"] = self.vocab_size
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1_seq1"] = self.mag[n_train:(n_train+n_test)]
        test_data["x1_seq2"] = self.mag_keywords[n_train:(n_train+n_test)]
        test_data["x2_seq1"] = self.aminer[n_train:(n_train+n_test)]
        test_data["x2_seq2"] = self.aminer_keywords[n_train:(n_train+n_test)]
        test_data["y"] = self.labels[n_train:(n_train+n_test)]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1_seq1"] = self.mag[n_train+n_test:(n_train+n_test*2)]
        valid_data["x1_seq2"] = self.mag_keywords[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2_seq1"] = self.aminer[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2_seq2"] = self.aminer_keywords[n_train+n_test:(n_train+n_test*2)]
        valid_data["y"] = self.labels[n_train+n_test:(n_train+n_test*2)]
        print("valid labels", len(valid_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "author_rnn_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "author_rnn_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "author_rnn_valid.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default=settings.AUTHOR_DATA_DIR, help="Input file directory")
    parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
    parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
    parser.add_argument('--train-num', type=int, default=800, help='Training size.')
    parser.add_argument('--test-num', type=int, default=200, help='Testing size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
    parser.add_argument('--max-sequence-length', type=int, default=17,
                        help="Max sequence length for raw sequences")
    parser.add_argument('--max-key-sequence-length', type=int, default=8,
                        help="Max key sequence length for key sequences")
    args = parser.parse_args()
    dataset = CNNMatchDataset(file_dir=args.file_dir, matrix_size1=args.matrix_size1, matrix_size2=args.matrix_size2, seed=args.seed, shuffle=True, args=args, use_emb=False)
    # dataset = AuthorRNNMatchDataset(args.file_dir, args.max_sequence_length,
    #                           args.max_key_sequence_length, shuffle=True, seed=args.seed, args=args)