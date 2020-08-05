from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
import os
from os.path import join
import numpy as np
import sklearn
from torch.utils.data import Dataset

from utils import feature_utils
from utils import data_utils
from utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class AffCNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle, args):

        self.file_dir = file_dir

        self.matrix_size_1_long = matrix_size1
        self.matrix_size_2_short = matrix_size2

        # load training pairs
        pos_pairs = data_utils.load_json(file_dir, 'train_positive_affi.json')
        pos_pairs = [(p['aminer_affi'], p['mag_affi']) for p in pos_pairs]
        neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi.json')
        neg_pairs = [(p['aminer_affi'], p['mag_affi']) for p in neg_pairs]
        n_pos = len(pos_pairs)
        labels = [1] * len(pos_pairs) + [0] * len(pos_pairs)
        pairs = pos_pairs + [neg_pairs[x] for x in range(n_pos)]  # label balanced is important

        n_matrix = len(pairs)
        self.X_long = np.zeros((n_matrix, self.matrix_size_1_long, self.matrix_size_1_long))
        self.X_short = np.zeros((n_matrix, self.matrix_size_2_short, self.matrix_size_2_short))
        self.Y = np.zeros(n_matrix, dtype=np.long)
        count = 0
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d', i)
            item_a, item_m = pair
            cur_y = labels[i]
            matrix1 = self.sentences_long_to_matrix(item_a['name'], item_m['DisplayName'])
            self.X_long[count] = feature_utils.scale_matrix(matrix1)
            # matrix2 = self.sentences_short_to_matrix(item_a['main_body'], item_m['NormalizedName'])
            matrix2 = self.sentences_short_to_matrix_2(item_a['name'], item_m['DisplayName'])
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

        self.N = len(self.Y)

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
        twords1 = feature_utils.get_words(title1)[: self.matrix_size_2_short]
        twords2 = feature_utils.get_words(title2)[: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix

    def sentences_short_to_matrix_2(self, title1, title2):
        title1 = title1.split()
        title2 = title2.split()
        # print(title1)
        overlap = set(title1).intersection(title2)
        new_seq_mag = []
        new_seq_aminer = []
        for w in title1:
            if w in overlap:
                new_seq_mag.append(w)
        for w in title2:
            if w in overlap:
                new_seq_aminer.append(w)

        twords1 = new_seq_mag[: self.matrix_size_2_short]
        twords2 = new_seq_aminer[: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                matrix[i][j] = (1 if word1 == word2 else -1)
        return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default=settings.AFF_DATA_DIR, help="Input file directory")
    parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
    parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
    parser.add_argument('--train-num', type=int, default=800, help='Training size.')
    parser.add_argument('--test-num', type=int, default=200, help='Testing size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
    args = parser.parse_args()
    dataset = AffCNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.seed, shuffle=args.shuffle, args=args)