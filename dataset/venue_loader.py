from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from os.path import join
import sklearn
import json
import codecs
import pickle
import _pickle
import nltk
import argparse
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text
from keras.preprocessing.sequence import pad_sequences

from utils import feature_utils
from utils import data_utils
from utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class VenueCNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle, args):

        self.file_dir = file_dir

        self.matrix_size_1_long = matrix_size1
        self.matrix_size_2_short = matrix_size2

        self.train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train.txt'), 'r'))
        self.mag = [nltk.word_tokenize(p[1]) for p in self.train_data]
        self.aminer = [nltk.word_tokenize(p[2]) for p in self.train_data]
        self.labels = [p[0] for p in self.train_data]

        self.calc_keyword_seqs()

        n_matrix = len(self.labels)
        self.X_long = np.zeros((n_matrix, self.matrix_size_1_long, self.matrix_size_1_long))
        self.X_short = np.zeros((n_matrix, self.matrix_size_2_short, self.matrix_size_2_short))
        self.Y = np.zeros(n_matrix, dtype=np.long)
        count = 0
        for i, cur_y in enumerate(self.labels):
            if i % 100 == 0:
                logger.info('pairs to matrices %d', i)
            v1 = self.mag[i]
            v1 = " ".join([str(v) for v in v1])
            v2 = self.aminer[i]
            v2 = " ".join([str(v) for v in v2])
            v1_key = self.mag_venue_keywords[i]
            v1_key = " ".join([str(v) for v in v1_key])
            v2_key = self.aminer_venue_keywords[i]
            v2_key = " ".join([str(v) for v in v2_key])
            matrix1 = self.sentences_long_to_matrix(v1, v2)
            self.X_long[count] = feature_utils.scale_matrix(matrix1)
            matrix2 = self.sentences_short_to_matrix(v1_key, v2_key)
            self.X_short[count] = feature_utils.scale_matrix(matrix2)
            self.Y[count] = cur_y
            count += 1

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

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "venue_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "venue_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "venue_valid.pkl")


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X_long[idx], self.X_short[idx], self.Y[idx]

    def sentences_long_to_matrix(self, title1, title2):
        twords1 = feature_utils.get_words(title1)[: self.matrix_size_1_long]
        twords2 = feature_utils.get_words(title2)[: self.matrix_size_1_long]

        matrix = -np.ones((self.matrix_size_1_long, self.matrix_size_1_long))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def sentences_short_to_matrix(self, title1, title2):
        # print("short---", title1, "v.s.", title2)
        twords1 = feature_utils.get_words(title1)[: self.matrix_size_2_short]
        twords2 = feature_utils.get_words(title2)[: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

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
        self.mag_venue_keywords = mag_keywords
        self.aminer_venue_keywords = aminer_keywords


class VenueRNNMatchDataset(Dataset):

    def __init__(self, file_dir, max_seq1_len, max_seq2_len, shuffle, seed, args):

        self.max_seq1_len = max_seq1_len
        self.max_seq2_len = max_seq2_len
        self.train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train.txt'), 'r'))
        # self.tokenizer = _pickle.load(open(join(settings.DATA_DIR, 'venues', "tokenizer"), "rb"))
        # print(self.tokenizer)

        # corpus = []
        # for item in self.train_data:
        #     corpus.append(item[1].lower())
        #     corpus.append(item[2].lower())
        # vectorizer = CountVectorizer()
        # X = vectorizer.fit_transform(corpus)
        # print(len(vectorizer.vocabulary_), vectorizer.vocabulary_)
        # t = Tokenizer()
        # t.fit_on_texts(corpus)

        t = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        self.vocab_size = len(t.word_counts)
        print("vocab size", self.vocab_size)
        print("tokenizer", t.word_counts, t.word_index)
        self.load_stop_words()  #TODO

        # self.mag = [nltk.word_tokenize(p[1].lower()) for p in self.train_data]
        # self.aminer = [nltk.word_tokenize(p[2].lower()) for p in self.train_data]
        self.mag = t.texts_to_sequences([p[1] for p in self.train_data])
        # print("mag", self.mag)
        self.aminer = t.texts_to_sequences([p[2] for p in self.train_data])
        self.labels = [p[0] for p in self.train_data]

        self.calc_keyword_seqs()
        self.mag = pad_sequences(self.mag, maxlen=self.max_seq1_len)
        self.aminer = pad_sequences(self.aminer, maxlen=self.max_seq1_len)
        self.mag_venue_keywords = pad_sequences(self.mag_venue_keywords, maxlen=self.max_seq2_len)
        self.aminer_venue_keywords = pad_sequences(self.aminer_venue_keywords, maxlen=max_seq2_len)

        if shuffle:
            self.mag, self.aminer, self.mag_venue_keywords, self.aminer_venue_keywords, self.labels = sklearn.utils.shuffle(
                self.mag, self.aminer, self.mag_venue_keywords, self.aminer_venue_keywords, self.labels,
                random_state=seed
            )

        self.N = len(self.labels)

        n_train = args.train_num
        n_test = args.test_num

        train_data = {}
        train_data["x1_seq1"] = self.mag[:n_train]
        train_data["x1_seq2"] = self.mag_venue_keywords[:n_train]
        train_data["x2_seq1"] = self.aminer[:n_train]
        train_data["x2_seq2"] = self.aminer_venue_keywords[:n_train]
        train_data["y"] = self.labels[:n_train]
        train_data["vocab_size"] = self.vocab_size
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1_seq1"] = self.mag[n_train:(n_train+n_test)]
        test_data["x1_seq2"] = self.mag_venue_keywords[n_train:(n_train+n_test)]
        test_data["x2_seq1"] = self.aminer[n_train:(n_train+n_test)]
        test_data["x2_seq2"] = self.aminer_venue_keywords[n_train:(n_train+n_test)]
        test_data["y"] = self.labels[n_train:(n_train+n_test)]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1_seq1"] = self.mag[n_train+n_test:(n_train+n_test*2)]
        valid_data["x1_seq2"] = self.mag_venue_keywords[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2_seq1"] = self.aminer[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2_seq2"] = self.aminer_venue_keywords[n_train+n_test:(n_train+n_test*2)]
        valid_data["y"] = self.labels[n_train+n_test:(n_train+n_test*2)]
        print("valid labels", len(valid_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "venue_rnn_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "venue_rnn_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "venue_rnn_valid.pkl")

    def load_stop_words(self):
        self.stop_list = []
        with codecs.open(join(settings.VENUE_DATA_DIR, 'stoplist.txt'), 'r', 'utf-8') as f:
            for word in f.readlines():
                self.stop_list.append(word[:-1])

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
        self.mag_venue_keywords = mag_keywords
        self.aminer_venue_keywords = aminer_keywords
        print("mag keywords", self.mag_venue_keywords)


class PairTextDataset(object):

    def __init__(self, file_dir, seed, shuffle, max_sequence_length, max_key_sequence_length, batch_size, multiple):
        self.file_dir = file_dir

        # load data
        logger.info('loading training pairs...')
        self.msl = max_sequence_length
        self.mksl = max_key_sequence_length
        self.train_data = json.load(codecs.open(join(settings.VENUE_DATA_DIR, 'train.txt'), 'r', 'utf-8'))
        # if shuffle:
        #     self.train_data = sklearn.utils.shuffle(self.train_data, random_state=37)
        self.tokenizer = _pickle.load(open(join(settings.DATA_DIR, 'venues', "tokenizer"), "rb"))
        self.vocab_size = self.split_and_tokenize()
        self.stop_list = []
        self.batch_size = batch_size
        with codecs.open(join(settings.VENUE_DATA_DIR, 'stoplist.txt'), 'r', 'utf-8') as f:
            for word in f.readlines():
                self.stop_list.append(word[:-1])
        self.labels = []
        self.mag = []
        self.aminer = []
        self.keyword_mag = []
        self.keyword_aminer = []
        self.length_mag = []
        self.length_aminer = []
        self.jaccard = []
        self.inverse_pairs = []
        self.mag = self.tokenizer.texts_to_sequences([p[1] for p in self.train_data])
        self.aminer = self.tokenizer.texts_to_sequences([p[2] for p in self.train_data])
        for i, pair in enumerate(self.train_data):
            len_mag, len_aminer, keyword_mag, keyword_aminer, jaccard, inverse_pairs = self.preprocess(self.mag[i],
                                                                                                       self.aminer[i])
            self.labels.append(pair[0])
            self.length_mag.append(len_mag)
            self.length_aminer.append(len_aminer)
            self.keyword_mag.append(keyword_mag)
            self.keyword_aminer.append(keyword_aminer)
            self.jaccard.append([np.float32(jaccard)] * (multiple * 2))
            self.inverse_pairs.append([np.float32(inverse_pairs)] * multiple)
        self.mag = pad_sequences(self.mag, maxlen=self.msl)
        self.aminer = pad_sequences(self.aminer, maxlen=self.msl)
        self.labels, self.mag, self.aminer, self.length_mag, self.length_aminer, self.keyword_mag, self.keyword_aminer, self.jaccard, self.inverse_pairs = np.array(
            self.labels), np.array(self.mag), np.array(self.aminer), np.array(self.length_mag), np.array(
            self.length_aminer), np.array(self.keyword_mag), np.array(self.keyword_aminer), np.array(
            self.jaccard), np.array(self.inverse_pairs)
        logger.info('training pairs loaded')

        self.n_pairs = len(self.labels)
        logger.info('all pairs count %d', self.n_pairs)

    def split_and_tokenize(self):
        for i, pair in enumerate(self.train_data.copy()):
            seq1 = text.text_to_word_sequence(pair[1])
            seq2 = text.text_to_word_sequence(pair[2])
            self.train_data[i] = [pair[0], seq1, seq2]
        return len(self.tokenizer.word_index)

    def preprocess(self, seq1, seq2, use_stop_word=False):
        overlap = set(seq1).intersection(seq2)
        jaccard = len(overlap) / (len(seq1) + len(seq2) - len(overlap))
        inverse_pairs, keyword_seq1, keyword_seq2 = self.compute_inverse_pairs(seq1, seq2, overlap)
        return len(seq1), len(seq2), keyword_seq1, keyword_seq2, jaccard, inverse_pairs

    def remove_stop_word(self, seq, stop_word=None):
        s = []
        stop_list = self.stop_list if not stop_word else stop_word
        for word in seq:
            if word not in stop_list:
                s.append(word)
        return [0] * (self.mksl - len(s)) + s if len(s) <= self.mksl else s[:self.mksl]

    def compute_inverse_pairs(self, seq1, seq2, overlap):
        look_up = {}
        new_seq1 = []
        new_seq2 = []
        for w in seq1:
            if w in overlap:
                look_up[w] = len(look_up) + 1
                new_seq1.append(look_up[w])
        for w in seq2:
            if w in overlap:
                new_seq2.append(look_up[w])
        result = 0
        for i in range(len(new_seq2)):
            for j in range(i, len(new_seq2)):
                if new_seq2[j] < i + 1:
                    result -= 1
        return result, \
               [0] * (self.mksl - len(new_seq1)) + new_seq1 if len(new_seq1) <= self.mksl else new_seq1[:self.mksl], \
               [0] * (self.mksl - len(new_seq2)) + new_seq2 if len(new_seq2) <= self.mksl else new_seq2[:self.mksl]

    def split_dataset(self, test_and_valid_size):
        train_ratio = 100 - test_and_valid_size
        valid_ratio = test_and_valid_size / 2
        N = len(self.labels)
        train_start, valid_start, test_start = \
            0, int(N * train_ratio / 100), int(N * (train_ratio + valid_ratio) / 100)
        train = {'mag': self.mag[train_start:valid_start], 'aminer': self.aminer[train_start:valid_start],
                 'keyword_mag': self.keyword_mag[train_start:valid_start],
                 'keyword_aminer': self.keyword_aminer[train_start:valid_start],
                 'jaccard': self.jaccard[train_start:valid_start],
                 'inverse': self.inverse_pairs[train_start:valid_start], 'labels': self.labels[train_start:valid_start]}
        valid = {'mag': self.mag[valid_start:test_start], 'aminer': self.aminer[valid_start:test_start],
                 'keyword_mag': self.keyword_mag[valid_start:test_start],
                 'keyword_aminer': self.keyword_aminer[valid_start:test_start],
                 'jaccard': self.jaccard[valid_start:test_start],
                 'inverse': self.inverse_pairs[valid_start:test_start], 'labels': self.labels[valid_start:test_start]}
        test = {'mag': self.mag[test_start:], 'aminer': self.aminer[test_start:],
                 'keyword_mag': self.keyword_mag[test_start:],
                 'keyword_aminer': self.keyword_aminer[test_start:],
                 'jaccard': self.jaccard[test_start:],
                 'inverse': self.inverse_pairs[test_start:], 'labels': self.labels[test_start:]}
        return DataLoader(self.batch_size, train), DataLoader(self.batch_size, valid), DataLoader(self.batch_size, test)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        return self.mag[idx], self.aminer[idx], self.jaccard[idx], self.keyword_mag[idx], self.keyword_aminer[idx], \
               self.inverse_pairs[idx], self.labels[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default=settings.VENUE_DATA_DIR, help="Input file directory")
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
    # dataset = VenueCNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.seed, shuffle=False, args=args)
    dataset = VenueRNNMatchDataset(args.file_dir, args.max_sequence_length,
                              args.max_key_sequence_length, shuffle=True, seed=args.seed, args=args)
