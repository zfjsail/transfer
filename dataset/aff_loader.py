from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
import os
from os.path import join
import numpy as np
import torch
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text
from keras.preprocessing.sequence import pad_sequences

from utils import feature_utils
from utils import data_utils
from utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class AffCNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle, args, use_emb=True):

        self.file_dir = file_dir

        self.matrix_size_1_long = matrix_size1
        self.matrix_size_2_short = matrix_size2

        self.use_emb = use_emb
        if self.use_emb:
            self.pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))
        self.tokenizer = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        # load training pairs
        # pos_pairs = data_utils.load_json(file_dir, 'train_positive_affi.json')
        # pos_pairs = [(p['aminer_affi'], p['mag_affi']) for p in pos_pairs]
        pos_pairs = data_utils.load_json(file_dir, "label_data_aff_zhoushao.json")[:600]
        pos_pairs = [({"name": p["affiliation"]}, {"DisplayName": p["label"]}) for p in pos_pairs if p["label"] != "[NIF]"]
        # neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi.json')
        neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi_clean.json')[:600]
        neg_pairs = [(p['aminer_affi'], p['mag_affi']) for p in neg_pairs]
        pairs_add = data_utils.load_json(file_dir, "mag_aminer_hard_correct_zfj_copy.json")
        print("add pairs", len(pairs_add))
        pos_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "1"]
        neg_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "0"]
        n_pos = len(pos_pairs)
        # labels = [1] * len(pos_pairs) + [0] * len(pos_pairs)
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        # pairs = pos_pairs + [neg_pairs[x] for x in range(n_pos)]  # label balanced is important
        pairs = pos_pairs + neg_pairs  # label balanced is important

        n_matrix = len(pairs)
        self.X_long = np.zeros((n_matrix, self.matrix_size_1_long, self.matrix_size_1_long))
        self.X_short = np.zeros((n_matrix, self.matrix_size_2_short, self.matrix_size_2_short))
        self.Y = np.zeros(n_matrix, dtype=np.long)
        count = 0
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                print('pairs to matrices', i)
            item_a, item_m = pair
            cur_y = labels[i]
            matrix1 = self.sentences_long_to_matrix(item_a['name'], item_m['DisplayName'])
            # print("matrix1", matrix1)
            # print(item_a['name'])
            # print(item_m['DisplayName'])
            self.X_long[count] = feature_utils.scale_matrix(matrix1)
            # matrix2 = self.sentences_short_to_matrix(item_a['main_body'], item_m['NormalizedName'])
            matrix2 = self.sentences_short_to_matrix_2(item_a['name'], item_m['DisplayName'])
            # print("matrix2", matrix2)
            self.X_short[count] = feature_utils.scale_matrix(matrix2)
            self.Y[count] = cur_y
            count += 1

            # # transpose
            # self.X_long[count] = feature_utils.scale_matrix(matrix1.transpose())
            # self.X_short[count] = feature_utils.scale_matrix(matrix2.transpose())
            # self.Y[count] = cur_y
            # count += 1

        print("shuffle", shuffle)
        if shuffle:
            self.X_long, self.X_short, self.Y = sklearn.utils.shuffle(
                self.X_long, self.X_short, self.Y,
                random_state=seed
            )

        self.N = len(self.Y)

        n_train = int(self.N*0.6)
        n_test = int(self.N*0.2)

        train_data = {}
        train_data["x1"] = self.X_long[:n_train]
        train_data["x2"] = self.X_short[:n_train]
        train_data["y"] = self.Y[:n_train]
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1"] = self.X_long[n_train:(n_train+n_test)]
        test_data["x2"] = self.X_short[n_train:(n_train+n_test)]
        test_data["y"] = self.Y[n_train:(n_train+n_test)]
        print("test labels", len(test_data["y"]), test_data["y"])

        valid_data = {}
        valid_data["x1"] = self.X_long[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2"] = self.X_short[n_train+n_test:(n_train+n_test*2)]
        valid_data["y"] = self.Y[n_train+n_test:(n_train+n_test*2)]
        print("valid labels", len(valid_data["y"]), valid_data["y"])

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "aff_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "aff_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "aff_valid.pkl")


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_long[idx], self.X_short[idx], self.Y[idx]

    def sentences_long_to_matrix(self, title1, title2):

        if self.use_emb:
            twords1 = self.tokenizer.texts_to_sequences([title1])[0][: self.matrix_size_1_long]
            twords2 = self.tokenizer.texts_to_sequences([title2])[0][: self.matrix_size_1_long]
        else:
            twords1 = feature_utils.get_words(title1)[: self.matrix_size_1_long]
            twords2 = feature_utils.get_words(title2)[: self.matrix_size_1_long]
        # print("twords1", twords1)
        # print("twords2", twords2)

        matrix = -np.ones((self.matrix_size_1_long, self.matrix_size_1_long))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                v = -1
                if word1 == word2:
                    v = 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[word1].reshape(1, -1),
                                          self.pretrain_emb[word2].reshape(1, -1))[0][0]
                    # print("cos", v)
                matrix[i][j] = v
        return matrix

    def sentences_short_to_matrix(self, title1, title2):
        # twords1 = feature_utils.get_words(title1)[: self.matrix_size_2_short]
        # twords2 = feature_utils.get_words(title2)[: self.matrix_size_2_short]

        twords1 = self.tokenizer.texts_to_sequences([title1])[0][: self.matrix_size_2_short]
        twords2 = self.tokenizer.texts_to_sequences([title2])[0][: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def sentences_short_to_matrix_2(self, title1, title2):

        if self.use_emb:
            twords1 = self.tokenizer.texts_to_sequences([title1])[0]
            twords2 = self.tokenizer.texts_to_sequences([title2])[0]
        else:
            twords1 = title1.split()
            twords2 = title2.split()

        # title1 = title1.split()
        # title2 = title2.split()
        # print(title1)
        overlap = set(twords1).intersection(twords2)
        new_seq_mag = []
        new_seq_aminer = []
        for w in twords1:
            if w in overlap:
                new_seq_mag.append(w)
        for w in twords2:
            if w in overlap:
                new_seq_aminer.append(w)

        twords1 = new_seq_mag[: self.matrix_size_2_short]
        twords2 = new_seq_aminer[: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                v = -1
                if word1 == word2:
                    v = 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[word1].reshape(1, -1),
                                          self.pretrain_emb[word2].reshape(1, -1))[0][0]
                    # print("cos", v)
                matrix[i][j] = v
                # matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix


class AffRNNMatchDataset(Dataset):

    def __init__(self, file_dir, max_seq1_len, max_seq2_len, shuffle, seed, args):

        self.max_seq1_len = max_seq1_len
        self.max_seq2_len = max_seq2_len

        # load training pairs
        # pos_pairs = data_utils.load_json(file_dir, 'train_positive_affi.json')
        # pos_pairs = [(p['aminer_affi'], p['mag_affi']) for p in pos_pairs]
        # neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi.json')
        # neg_pairs = [(p['aminer_affi'], p['mag_affi']) for p in neg_pairs]
        # n_pos = len(pos_pairs)
        pos_pairs = data_utils.load_json(file_dir, "label_data_aff_zhoushao.json")[:600]
        pos_pairs = [({"name": p["affiliation"]}, {"DisplayName": p["label"]}) for p in pos_pairs if p["label"] != "[NIF]"]
        # neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi.json')
        neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi_clean.json')[:600]
        neg_pairs = [(p['aminer_affi'], p['mag_affi']) for p in neg_pairs]
        pairs_add = data_utils.load_json(file_dir, "mag_aminer_hard_correct_zfj_copy.json")
        print("add pairs", len(pairs_add))
        pos_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "1"]
        neg_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "0"]

        self.labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        pairs = pos_pairs + neg_pairs  # label balanced is important

        # corpus = []
        # for item in pairs:
        #     corpus.append(item[0]["name"].lower())
        #     corpus.append(item[1]["DisplayName"].lower())
        #
        # t = Tokenizer(num_words=9999)
        # t.fit_on_texts(corpus)

        t = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        self.vocab_size = len(t.word_counts)
        print("vocab size", self.vocab_size)
        # print("tokenizer", t.word_index)

        self.mag = t.texts_to_sequences([p[1]["DisplayName"] for p in pairs])
        for mag_aff in self.mag:
            for word_idx in mag_aff:
                assert word_idx <= 100000
        self.aminer = t.texts_to_sequences([p[0]["name"] for p in pairs])
        self.mag = pad_sequences(self.mag, maxlen=self.max_seq1_len)
        self.aminer = pad_sequences(self.aminer, maxlen=self.max_seq1_len)

        # self.mag_keywords = t.texts_to_sequences([p[1]["NormalizedName"] for p in pairs])
        # self.aminer_keywords = t.texts_to_sequences([p[0]["main_body"] for p in pairs])
        self.calc_keyword_seqs()
        # self.mag_keywords = t.texts_to_sequences(self.mag_keywords)
        # self.aminer_keywords = t.texts_to_sequences(self.aminer_keywords)

        self.mag_keywords = pad_sequences(self.mag_keywords, maxlen=max_seq2_len)
        self.aminer_keywords = pad_sequences(self.aminer_keywords, maxlen=max_seq2_len)

        if shuffle:
            self.mag, self.aminer, self.mag_keywords, self.aminer_keywords, self.labels = sklearn.utils.shuffle(
                self.mag, self.aminer, self.mag_keywords, self.aminer_keywords, self.labels,
                random_state=seed
            )

        self.N = len(self.labels)

        n_train = int(self.N * 0.6)
        n_test = int(self.N * 0.2)

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
        data_utils.dump_large_obj(train_data, out_dir, "aff_rnn_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "aff_rnn_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "aff_rnn_valid.pkl")

    def calc_keyword_seqs(self):
        N = len(self.mag)
        mag_keywords = []
        aminer_keywords = []
        for i in range(N):
            cur_v_mag = self.mag[i]
            cur_v_aminer = self.aminer[i]
            overlap = set(cur_v_mag).intersection(cur_v_aminer)
            new_seq_mag = []
            new_seq_aminer = []
            for w in cur_v_mag:
                if w in overlap:
                    new_seq_mag.append(w)
            for w in cur_v_aminer:
                if w in overlap:
                    new_seq_aminer.append(w)
            mag_keywords.append(new_seq_mag)
            aminer_keywords.append(new_seq_aminer)
        self.mag_keywords = mag_keywords
        self.aminer_keywords = aminer_keywords
        print("mag keywords", self.mag_keywords)


def filter_aff_neg_pairs():
    neg_pairs = data_utils.load_json(settings.AFF_DATA_DIR, 'train_negative_affi.json')
    neg_pairs_cleaned = []
    for i, pair in enumerate(neg_pairs):
        if i % 100 == 0:
            print("pair", i)
        mag_aff = pair["mag_affi"]
        aminer_aff = pair["aminer_affi"]
        aff1 = mag_aff["NormalizedName"].split()
        aff2 = aminer_aff["main_body"].split()
        common = set(aff1).intersection(aff2)
        if len(common) > 1:
            neg_pairs_cleaned.append(pair)
    print("after cleaned", len(neg_pairs_cleaned))
    data_utils.dump_json(neg_pairs_cleaned, settings.AFF_DATA_DIR, "train_negative_affi_clean.json")


def filter_hard_aff_pairs():
    df = pd.read_excel(join(settings.AFF_DATA_DIR, "mag_aminer_1（人工标注）.xls"))
    # print(df)
    pairs = []
    for index, row in df.iterrows():
        mag_aff2 = row["mag_NormalizedName"].lower()
        aminer_aff2 = row["aminer_main_body"].lower()
        label = row["label"]
        common = set(aminer_aff2.split()).intersection(mag_aff2.split())
        if len(common) > 0:
            print(mag_aff2, "---", aminer_aff2, label)
            mag_aff1 = row["mag_DisplayName"].lower()
            aminer_aff1 = row["aminer_org_name"].lower()
            mag_id = row["mag_id"]
            aminer_id = row["aminer_id"]
            cur_dict = {
                "mag_affi": {
                    "id": mag_id,
                    "DisplayName": mag_aff1,
                    "NormalizedName": mag_aff2
                },
                "aminer_affi": {
                    "id": aminer_id,
                    "name": aminer_aff1,
                    "main_body": aminer_aff2
                },
                "label": label,
                "label_zfj": ""
            }
            pairs.append(cur_dict)

    print("n_pairs", len(pairs))
    data_utils.dump_json(pairs, settings.AFF_DATA_DIR, "mag_aminer_hard_correct_zfj.json")


def check_labeled_zfj():
    pairs = data_utils.load_json(settings.AFF_DATA_DIR, "mag_aminer_hard_correct_zfj_copy.json")
    n_label_zfj = 0
    for pair in pairs:
        if pair["label_zfj"]:
            n_label_zfj += 1
    print("labeled until now", n_label_zfj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default=settings.AFF_DATA_DIR, help="Input file directory")
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
    # dataset = AffCNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.seed, shuffle=args.shuffle, args=args, use_emb=False)
    dataset = AffRNNMatchDataset(args.file_dir, args.max_sequence_length,
                              args.max_key_sequence_length, shuffle=True, seed=args.seed, args=args)
    # filter_aff_neg_pairs()
    # filter_hard_aff_pairs()
    # check_labeled_zfj()
