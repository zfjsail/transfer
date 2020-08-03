from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
from os.path import join
import os
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from dataset import ProcessedCNNInputDataset
from models.cnn import CNNMatchModel
from utils.data_utils import ChunkSampler
from utils import feature_utils
from utils import eval_utils
from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
parser.add_argument('--mat1-channel1', type=int, default=8, help='Matrix1 number of channels1.')
parser.add_argument('--mat1-kernel-size1', type=int, default=3, help='Matrix1 kernel size1.')
parser.add_argument('--mat1-channel2', type=int, default=16, help='Matrix1 number of channel2.')
parser.add_argument('--mat1-kernel-size2', type=int, default=2, help='Matrix1 kernel size2.')
parser.add_argument('--mat1-hidden', type=int, default=512, help='Matrix1 hidden dim.')
parser.add_argument('--mat2-channel1', type=int, default=8, help='Matrix2 number of channels1.')
parser.add_argument('--mat2-kernel-size1', type=int, default=2, help='Matrix2 kernel size1.')
parser.add_argument('--mat2-hidden', type=int, default=512, help='Matrix2 hidden dim')
parser.add_argument('--build-index-window', type=int, default=5, help='Matrix2 hidden dim')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--initial-accumulator-value', type=float, default=0.01, help='Initial accumulator value.')
parser.add_argument('--weight-decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="32,8",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8,1",
                    help="Heads in each layer, splitted with comma")
parser.add_argument('--batch', type=int, default=32, help="Batch size")
parser.add_argument('--dim', type=int, default=64, help="Embedding dimension")
parser.add_argument('--check-point', type=int, default=2, help="Check point")
parser.add_argument('--n-type-nodes', type=int, default=3, help="the number of different types of nodes")
parser.add_argument('--instance-normalization', action='store_true', default=True,
                    help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--entity-type', type=str, default="paper", help="Types of entities to match")

parser.add_argument('--file-dir', type=str, default=settings.AFF_DATA_DIR, help="Input file directory")
parser.add_argument('--alpha', type=float, default=0.2, help="Alpha for the leaky_relu.")
parser.add_argument('--train-ratio', type=float, default=10, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=10, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                    help="Adjust weights inversely proportional"
                         " to class frequencies in the input data")
parser.add_argument('--use-vertex-feature', action='store_true', default=False,
                    help="Whether to use vertices' structural features")
parser.add_argument('--sequence-size', type=int, default=16,
                    help="Sequence size (only useful for pscn)")
parser.add_argument('--neighbor-size', type=int, default=5,
                    help="Neighborhood size (only useful for pscn)")

args = parser.parse_args()


def train(epoch, train_loader, valid_loader, test_loader, model, optimizer, args=args):
    model.train()

    loss = 0.
    total = 0.

    for i_batch, batch in enumerate(train_loader):
        X_title, X_author, Y = batch
        # print("x1", X_title)
        # print("x2", X_author)
        # print("y", Y)
        # print(Y)
        bs = Y.shape[0]

        if args.cuda:
            X_title = X_title.cuda()
            X_author = X_author.cuda()
            Y = Y.cuda()

        optimizer.zero_grad()
        output, hidden = model(X_title.float(), X_author.float())
        # print("hidden", hidden)

        loss_train = F.nll_loss(output, Y.long())
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss epoch %d: %f", epoch, loss / total)

    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        best_thr = evaluate(epoch, valid_loader, model, return_best_thr=True, args=args)
        logger.info('eval on test data!...')
        evaluate(epoch, test_loader, model, thr=best_thr, args=args)


def evaluate(epoch, loader, model, thr=None, return_best_thr=False, args=args):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []

    for i_batch, batch in enumerate(loader):
        X_title, X_author, Y = batch
        bs = len(Y)

        if args.cuda:
            X_title = X_title.cuda()
            X_author = X_author.cuda()
            Y = Y.cuda()

        output, _ = model(X_title.float(), X_author.float())
        loss_batch = F.nll_loss(output, Y.long())
        loss += bs * loss_batch.item()

        y_true += Y.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                loss / total, auc, prec, rec, f1)

    if return_best_thr:  # valid
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None


def main(args=args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # dataset = AffCNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.build_index_window, args.seed, args.shuffle)
    dataset = ProcessedCNNInputDataset(args.entity_type, "train")
    dataset_test = ProcessedCNNInputDataset(args.entity_type, "test")
    N = len(dataset)
    N_test = len(dataset_test)
    # train_start, valid_start, test_start = \
    #     0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
    train_start = 0
    valid_start = int(N * 4 / 5)
    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(valid_start - train_start, 0))
    valid_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(N - valid_start, valid_start))
    test_loader = DataLoader(dataset, batch_size=args.batch,
                             sampler=ChunkSampler(N_test, 0))
    model = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                          mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                          mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                          mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                          mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    model = model.float()

    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr,
                              initial_accumulator_value=args.initial_accumulator_value,
                              weight_decay=args.weight_decay)
    t_total = time.time()
    logger.info("training...")

    model.load_state_dict(torch.load(join(settings.OUT_VENUE_DIR, "venue-matching-cnn.mdl")))
    evaluate(0, test_loader, model, thr=None, args=args)
    for epoch in range(args.epochs):
        train(epoch, train_loader, valid_loader, test_loader, model, optimizer, args=args)

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    model_dir = join(settings.OUT_DIR, 'aff')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), join(model_dir, 'paper-matching-cnn.mdl'))
    logger.info('paper matching CNN model saved')

    # evaluate(args.epochs, test_loader, model, thr=best_thr, args=args)


if __name__ == '__main__':
    main(args=args)
