import sys, os, glob
import argparse
import time
import random
from copy import copy, deepcopy
from termcolor import colored, cprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

sys.path.append('../')
from msda_src.model_utils import get_model_class, get_critic_class
from msda_src.model_utils.domain_critic import ClassificationD, MMD, CoralD, WassersteinD
from msda_src.utils.io import AmazonDataset, AmazonDomainDataset
from msda_src.utils.io import say
from msda_src.utils.op import softmax

from dataset import ProcessedCNNInputDataset, ProcessedRNNInputDataset
from models.cnn import CNNMatchModel
from models.rnn import BiLSTM

from utils import settings

argparser = argparse.ArgumentParser(description="Learning to Adapt from Multi-Source Domains")
argparser.add_argument("--cuda", action="store_true")
argparser.add_argument("--train", type=str, default="aff,author,paper",
                       help="multi-source domains for training, separated with (,)")
argparser.add_argument("--test", type=str, default="venue",
                       help="target domain for testing")
argparser.add_argument("--eval_only", action="store_true")
argparser.add_argument("--critic", type=str, default="mmd")
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--batch_size_d", type=int, default=32)
argparser.add_argument("--max_epoch", type=int, default=200)
argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--lr_d", type=float, default=1e-4)
argparser.add_argument("--lambda_critic", type=float, default=0)
argparser.add_argument("--lambda_gp", type=float, default=0)
argparser.add_argument("--lambda_moe", type=float, default=1)
argparser.add_argument("--m_rank", type=int, default=10)
argparser.add_argument("--lambda_entropy", type=float, default=0.0)
argparser.add_argument("--load_model", type=str)
argparser.add_argument("--save_model", type=str)
argparser.add_argument("--base_model", type=str, default="rnn")
argparser.add_argument("--metric", type=str, default="mahalanobis",
                       help="mahalanobis: mahalanobis distance; biaffine: biaffine distance")

argparser.add_argument('--embedding-size', type=int, default=128,
                    help="Embeding size for LSTM layer")
argparser.add_argument('--hidden-size', type=int, default=32,
                    help="Hidden size for LSTM layer")
argparser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
argparser.add_argument('--max-vocab-size', type=int, default=100000, help="Maximum of Vocab Size")
argparser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
argparser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
argparser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
argparser.add_argument('--mat1-channel1', type=int, default=8, help='Matrix1 number of channels1.')
argparser.add_argument('--mat1-kernel-size1', type=int, default=3, help='Matrix1 kernel size1.')
argparser.add_argument('--mat1-channel2', type=int, default=16, help='Matrix1 number of channel2.')
argparser.add_argument('--mat1-kernel-size2', type=int, default=2, help='Matrix1 kernel size2.')
argparser.add_argument('--mat1-hidden', type=int, default=512, help='Matrix1 hidden dim.')
argparser.add_argument('--mat2-channel1', type=int, default=8, help='Matrix2 number of channels1.')
argparser.add_argument('--mat2-kernel-size1', type=int, default=2, help='Matrix2 kernel size1.')
argparser.add_argument('--mat2-hidden', type=int, default=512, help='Matrix2 hidden dim')
argparser.add_argument('--build-index-window', type=int, default=5, help='Matrix2 hidden dim')
argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
argparser.add_argument('--seed-delta', type=int, default=0, help='Random seed.')

argparser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (L2 loss on parameters).')
argparser.add_argument('--check-point', type=int, default=2, help="Check point")
argparser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")

args, _ = argparser.parse_known_args()

writer = SummaryWriter('runs/{}_sup_base_{}_moe_{}'.format(args.test, args.base_model, args.seed_delta))


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        # b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x * torch.log(x)
        b = -1.0 * b.sum()
        return b


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()

    def forward(self, x):
        return torch.norm(x, 1, 1).sum()


def domain_encoding(loaders, args, encoders):
    ''' Compute the encoding of domains, each domain is represented as its mean vector
        Note: the covariance inverse matrix is learned
    '''
    statistics = []

    for load_i, loader in enumerate(loaders):
        ind = 0
        labels = None
        S = []
        if args.base_model == "cnn":
            for batch1, batch2, label in loader:
                if args.cuda:
                    batch1 = Variable(batch1.cuda())
                    batch2 = Variable(batch2.cuda())
                _, s_out = encoders[load_i](batch1, batch2)

                S.append(s_out)
                if ind == 0:
                    labels = label
                else:
                    labels = torch.cat((labels, label), dim=0)
                ind += 1
        elif args.base_model == "rnn":
            for batch1, batch2, batch3, batch4, label in loader:
                if args.cuda:
                    batch1 = Variable(batch1.cuda())
                    batch2 = Variable(batch2.cuda())
                    batch3 = Variable(batch3.cuda())
                    batch4 = Variable(batch4.cuda())
                _, s_out = encoders[load_i](batch1, batch2, batch3, batch4)

                S.append(s_out)
                if ind == 0:
                    labels = label
                else:
                    labels = torch.cat((labels, label), dim=0)
                ind += 1
        else:
            raise NotImplementedError

        S = torch.cat(S, 0)
        neg_index = ((labels == 0).nonzero())
        pos_index = ((labels == 1).nonzero())
        neg_index = Variable(neg_index.expand(neg_index.size(0), S.size(1)))
        pos_index = Variable(pos_index.expand(pos_index.size(0), S.size(1)))
        if args.cuda:
            pos_index = pos_index.cuda()
            neg_index = neg_index.cuda()

        pos_S = torch.gather(S, 0, pos_index)
        neg_S = torch.gather(S, 0, neg_index)
        pos_mu_S = torch.mean(pos_S, dim=0, keepdim=True)
        neg_mu_S = torch.mean(neg_S, dim=0, keepdim=True)
        mu_S = torch.mean(S, dim=0, keepdim=True)

        statistics.append((mu_S, pos_mu_S, neg_mu_S))

    return statistics


TEMPERATURE = 4


def mahalanobis_metric_fast(p, mu, U, pos_mu, pos_U, neg_mu, neg_U):
    # covi = (cov + I).inverse()
    mahalanobis_distances = (p - mu).mm(U.mm(U.t())).mm((p - mu).t())
    pos_mahalanobis_distance = (p - pos_mu).mm(pos_U.mm(pos_U.t())).mm((p - pos_mu).t()).diag().sqrt().data
    neg_mahalanobis_distance = (p - neg_mu).mm(neg_U.mm(neg_U.t())).mm((p - neg_mu).t()).diag().sqrt().data
    mahalanobis_ratio1 = pos_mahalanobis_distance - neg_mahalanobis_distance
    mahalanobis_ratio2 = neg_mahalanobis_distance - pos_mahalanobis_distance
    max_ratio = torch.max(mahalanobis_ratio1, mahalanobis_ratio2)

    return max_ratio  # / TEMPERATURE
    # return mahalanobis_distances.diag().sqrt().data


def mahalanobis_metric(p, S, L, U, pos_U, neg_U, args, encoder=None):
    r''' Compute the mahalanobis distance between the encoding of a sample (p) and a set (S).
    Args:
        p: tensor (batch_size, dim), a batch of samples
        S: tensor (size, dim), a domain which contains a set of samples
        encoder: a module used for encoding p and S
    Return:
        mahalanobis_distances: tensor (batch_size)
    '''

    if encoder is not None:
        p = encoder(p)  # (batch_size, dim)
        S = encoder(S)  # (size, dim)

    neg_index = ((L == 0).nonzero())
    pos_index = ((L == 1).nonzero())

    neg_index = neg_index.expand(neg_index.size(0), S.data.size(1))
    pos_index = pos_index.expand(pos_index.size(0), S.data.size(1))

    neg_S = torch.gather(S, 0, neg_index)
    pos_S = torch.gather(S, 0, pos_index)
    neg_mu = torch.mean(neg_S, dim=0, keepdim=True)
    pos_mu = torch.mean(pos_S, dim=0, keepdim=True)

    pos_mahalanobis_distance = (p - pos_mu).mm(pos_U.mm(pos_U.t())).mm((p - pos_mu).t()).diag().sqrt()
    neg_mahalanobis_distance = (p - neg_mu).mm(neg_U.mm(neg_U.t())).mm((p - neg_mu).t()).diag().sqrt()

    mahalanobis_ratio1 = pos_mahalanobis_distance - neg_mahalanobis_distance
    mahalanobis_ratio2 = neg_mahalanobis_distance - pos_mahalanobis_distance

    max_ratio = torch.max(mahalanobis_ratio1, mahalanobis_ratio2)

    return max_ratio.clamp(0.01, 2)  # / TEMPERATURE # .clamp(0.001, 1)

    # mu_S = torch.mean(S, dim=0, keepdim=True) # (1, dim)
    # mahalanobis_distances = (p - mu_S).mm(U.mm(U.t())).mm((p - mu_S).t())
    # return mahalanobis_distances.diag().sqrt().clamp(0.01, 2)


def biaffine_metric_fast(p, mu, U):
    biaffine_distances = p.mm(U).mm(mu.t())
    return biaffine_distances.squeeze(1).data


def biaffine_metric(p, S, U, W, V, args, encoder=None):
    ''' Compute the biaffine distance between the encoding of a sample (p) and a set (S).
    Args:
        p: tensor (batch_size, dim), a batch of samples
        U: matrix (dim, dim)
        S: tensor (size, dim), a domain which contains a set of samples
        encoder: a module used for encoding p and S
    Return:
        biaffine_distance: tensor (batch_size)
    '''

    if encoder is not None:
        p = encoder(p)
        S = encoder(S)

    mu_S = torch.mean(S, dim=0, keepdim=True)
    biaffine_distances = p.mm(U).mm(mu_S.t()) + p.mm(W) + mu_S.mm(V)  # extra components
    return biaffine_distances.squeeze(1).clamp(-10, 10)


DATA_DIR = "../../msda-data/amazon/chen12"


def train_epoch(iter_cnt, encoders, classifiers, critic, mats, data_loaders, args, optim_model, epoch):
    encoders, encoder_dst = encoders
    map(lambda m: m.train(), [critic, encoder_dst] + classifiers + encoders)

    train_loaders, train_loader_dst, valid_loader = data_loaders
    dup_train_loaders = deepcopy(train_loaders)

    # mtl_criterion = nn.CrossEntropyLoss()
    moe_criterion = nn.NLLLoss()  # with log_softmax separated
    kl_criterion = nn.MSELoss()
    entropy_criterion = HLoss()

    if args.metric == "biaffine":
        metric = biaffine_metric
        Us, Ws, Vs = mats
    else:
        metric = mahalanobis_metric
        Us, Ps, Ns = mats

    loss_total = 0
    n_batch = 0

    for batches, unl_batch in zip(zip(*train_loaders), train_loader_dst):
        if args.base_model == "cnn":
            train_batches1, train_batches2, train_labels = zip(*batches)
            train_batch1_dst, train_batch2_dst, train_labels_dst = unl_batch
        elif args.base_model == "rnn":
            train_batches1, train_batches2, train_batches3, train_batches4, train_labels = zip(*batches)
            train_batch1_dst, train_batch2_dst, train_batch3_dst, train_batch4_dst, train_labels_dst = unl_batch
        else:
            raise NotImplementedError

        iter_cnt += 1
        n_batch += 1
        if args.cuda:
            train_batches1 = [batch.cuda() for batch in train_batches1]
            train_batches2 = [batch.cuda() for batch in train_batches2]
            train_labels = [label.cuda() for label in train_labels]

            train_batch1_dst = train_batch1_dst.cuda()
            train_batch2_dst = train_batch2_dst.cuda()
            train_labels_dst = train_labels_dst.cuda()

            if args.base_model == "rnn":
                train_batches3 = [batch.cuda() for batch in train_batches3]
                train_batches4 = [batch.cuda() for batch in train_batches4]
                train_batch3_dst = train_batch3_dst.cuda()
                train_batch4_dst = train_batch4_dst.cuda()

        # train_batches = [Variable(batch) for batch in train_batches]
        # train_labels = [Variable(label) for label in train_labels]
        # unl_critic_batch = Variable(unl_critic_batch)
        # unl_critic_label = Variable(unl_critic_label)

        outputs_dst_transfer = []
        for i in range(len(train_batches1)):
            if args.base_model == "cnn":
                _, cur_hidden = encoders[i](train_batch1_dst, train_batch2_dst)
            elif args.base_model == "rnn":
                _, cur_hidden = encoders[i](train_batch1_dst, train_batch2_dst, train_batch3_dst, train_batch4_dst)
            else:
                raise NotImplementedError

            cur_output = classifiers[i](cur_hidden)
            outputs_dst_transfer.append(cur_output)

        optim_model.zero_grad()
        # loss_mtl = []
        loss_moe = []
        # loss_kl = []
        loss_entropy = []
        # loss_dan = []

        if args.base_model == "cnn":
            _, hidden_dst = encoder_dst(train_batch1_dst, train_batch2_dst)
        elif args.base_model == "rnn":
            _, hidden_dst = encoder_dst(train_batch1_dst, train_batch2_dst, train_batch3_dst, train_batch4_dst)
        else:
            raise NotImplementedError

        ms_outputs = []  # (n_sources, n_classifiers)
        hiddens = []
        hidden_corresponding_labels = []
        # labels = []
        if args.base_model == "cnn":
            for i, (batch1, batch2, label) in enumerate(zip(train_batches1, train_batches2, train_labels)):
                _, hidden = encoders[i](batch1, batch2)
                outputs = []
                # create output matrix:
                #     - (i, j) indicates the output of i'th source batch using j'th classifier
                hiddens.append(hidden)
                for classifier in classifiers:
                    output = classifier(hidden)
                    outputs.append(output)
                ms_outputs.append(outputs)
                hidden_corresponding_labels.append(label)
                # multi-task loss
                # loss_mtl.append(mtl_criterion(ms_outputs[i][i], label))
                # labels.append(label)

                # if args.lambda_critic > 0:
                #     # critic_batch = torch.cat([batch, unl_critic_batch])
                #     critic_label = torch.cat([1 - unl_critic_label, unl_critic_label])
                #     # critic_label = torch.cat([1 - unl_critic_label] * len(train_batches) + [unl_critic_label])
                #
                #     if isinstance(critic, ClassificationD):
                #         critic_output = critic(torch.cat(hidden, encoder(unl_critic_batch)))
                #         loss_dan.append(critic.compute_loss(critic_output, critic_label))
                #     else:
                #         critic_output = critic(hidden, encoder(unl_critic_batch))
                #         loss_dan.append(critic_output)
                #
                #         # critic_output = critic(torch.cat(hiddens), encoder(unl_critic_batch))
                #         # loss_dan = critic_output
                # else:
                #     loss_dan = Variable(torch.FloatTensor([0]))
        elif args.base_model == "rnn":
            for i, (batch1, batch2, batch3, batch4, label) in enumerate(zip(
                    train_batches1, train_batches2, train_batches3, train_batches4, train_labels)):
                _, hidden = encoders[i](batch1, batch2, batch3, batch4)
                outputs = []
                # create output matrix:
                #     - (i, j) indicates the output of i'th source batch using j'th classifier
                hiddens.append(hidden)
                for classifier in classifiers:
                    output = classifier(hidden)
                    outputs.append(output)
                ms_outputs.append(outputs)
                hidden_corresponding_labels.append(label)
        else:
            raise NotImplementedError

        # assert (len(outputs) == len(outputs[0]))
        source_ids = range(len(train_batches1))
        # for i in source_ids:

        # support_ids = [x for x in source_ids if x != i]  # experts
        support_ids = [x for x in source_ids]  # experts

        # support_alphas = [ metric(
        #                      hiddens[i],
        #                      hiddens[j].detach(),
        #                      hidden_corresponding_labels[j],
        #                      Us[j], Ps[j], Ns[j],
        #                      args) for j in support_ids ]

        if args.metric == "biaffine":
            source_alphas = [metric(hidden_dst,
                                    hiddens[j].detach(),
                                    Us[0], Ws[0], Vs[0],  # for biaffine metric, we use a unified matrix
                                    args) for j in source_ids]
        else:
            source_alphas = [metric(hidden_dst,
                                    hiddens[j].detach(),
                                    hidden_corresponding_labels[j],
                                    Us[j], Ps[j], Ns[j],
                                    args) for j in source_ids]

        support_alphas = [source_alphas[x] for x in support_ids]

        # print torch.cat([ x.unsqueeze(1) for x in support_alphas ], 1)
        support_alphas = softmax(support_alphas)

        # meta-supervision: KL loss over \alpha and real source
        source_alphas = softmax(source_alphas)  # [ 32, 32, 32 ]
        source_labels = [torch.FloatTensor([x == len(train_batches1)]) for x in source_ids]  # one-hot
        if args.cuda:
            source_alphas = [alpha.cuda() for alpha in source_alphas]
            source_labels = [label.cuda() for label in source_labels]

        source_labels = Variable(torch.stack(source_labels, dim=0))  # 3*1
        source_alphas = torch.stack(source_alphas, dim=0)
        source_labels = source_labels.expand_as(source_alphas).permute(1, 0)
        source_alphas = source_alphas.permute(1, 0)
        # loss_kl.append(kl_criterion(source_alphas, source_labels))

        # entropy loss over \alpha
        # entropy_loss = entropy_criterion(torch.stack(support_alphas, dim=0).permute(1, 0))
        # print source_alphas
        loss_entropy.append(entropy_criterion(source_alphas))

        output_moe_i = sum([alpha.unsqueeze(1).repeat(1, 2) * F.softmax(outputs_dst_transfer[id], dim=1) \
                            for alpha, id in zip(support_alphas, support_ids)])
        # output_moe_full = sum([ alpha.unsqueeze(1).repeat(1, 2) * F.softmax(ms_outputs[i][id], dim=1) \
        #                         for alpha, id in zip(full_alphas, source_ids) ])

        loss_moe.append(moe_criterion(torch.log(output_moe_i), train_labels_dst))
        # loss_moe.append(moe_criterion(torch.log(output_moe_full), train_labels[i]))

        # loss_mtl = sum(loss_mtl)
        loss_moe = sum(loss_moe)
        # if iter_cnt < 400:
        #     lambda_moe = 0
        #     lambda_entropy = 0
        # else:
        lambda_moe = args.lambda_moe
        lambda_entropy = args.lambda_entropy
        # loss = (1 - lambda_moe) * loss_mtl + lambda_moe * loss_moe
        # loss = loss_mtl + lambda_moe * loss_moe
        loss = lambda_moe * loss_moe
        # loss_kl = sum(loss_kl)
        loss_entropy = sum(loss_entropy)
        loss += args.lambda_entropy * loss_entropy
        loss_total += loss.item()

        # if args.lambda_critic > 0:
        #     loss_dan = sum(loss_dan)
        #     loss += args.lambda_critic * loss_dan

        loss.backward()
        optim_model.step()

        if iter_cnt % 5 == 0:
            # [(mu_i, covi_i), ...]
            # domain_encs = domain_encoding(dup_train_loaders, args, encoder)
            if args.metric == "biaffine":
                mats = [Us, Ws, Vs]
            else:
                mats = [Us, Ps, Ns]

            # (curr_dev, oracle_curr_dev), confusion_mat = evaluate(
            #     encoder, classifiers,
            #     mats,
            #     [dup_train_loaders, valid_loader],
            #     args
            # )

            # say("\r" + " " * 50)
            # TODO: print train acc as well
            say("{} MOE loss: {:.4f}, Entropy loss: {:.4f}, "
                "loss: {:.4f}\n"
                # ", dev acc/oracle: {:.4f}/{:.4f}\n"
                .format(iter_cnt,
                        # loss_mtl.data[0],
                        loss_moe.item(),
                        loss_entropy.item(),
                        loss.data.item()
                        ))

    loss_total /= n_batch

    writer.add_scalar('training_loss',
                      loss_total,
                      epoch)

    say("\n")
    return iter_cnt


def compute_oracle(outputs, label, args):
    ''' Compute the oracle accuracy given outputs from multiple classifiers
    '''
    oracle = torch.ByteTensor([0] * label.shape[0])
    if args.cuda:
        oracle = oracle.cuda()
    for i, output in enumerate(outputs):
        pred = output.data.max(dim=1)[1]
        oracle |= pred.eq(label)
    return oracle


def evaluate(epoch, encoders, classifiers, mats, loaders, return_best_thrs, args, thr=None):
    ''' Evaluate model using MOE
    '''
    encoders, encoder_dst = encoders

    map(lambda m: m.eval(), [encoder_dst] + encoders + classifiers)

    if args.metric == "biaffine":
        Us, Ws, Vs = mats
    else:
        Us, Ps, Ns = mats

    source_loaders, valid_loader = loaders
    domain_encs = domain_encoding(source_loaders, args, encoders)

    oracle_correct = 0
    correct = 0
    tot_cnt = 0
    y_true = []
    y_pred = []
    y_score = []
    loss = 0.

    alpha_weights = np.zeros(shape=(len(encoders)))

    source_ids = range(len(domain_encs))
    cur_alpha_weights_stack = np.empty(shape=(0, len(domain_encs)))

    if args.base_model == "cnn":
        for batch1, batch2, label in valid_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                label = label.cuda()

            batch1 = Variable(batch1)
            batch2 = Variable(batch2)
            bs = len(batch1)

            # hidden = encoder(batch)

            _, hidden_dst = encoder_dst(batch1, batch2)

            # source_ids = range(len(domain_encs))
            if args.metric == "biaffine":
                alphas = [biaffine_metric_fast(hidden_dst, mu[0], Us[0]) \
                          for mu in domain_encs]
            else:
                alphas = [mahalanobis_metric_fast(hidden_dst, mu[0], U, mu[1], P, mu[2], N) \
                          for (mu, U, P, N) in zip(domain_encs, Us, Ps, Ns)]
            # alphas = [ (1 - x / sum(alphas)) for x in alphas ]
            alphas = softmax(alphas)
            if args.cuda:
                alphas = [alpha.cuda() for alpha in alphas]
            alphas = [Variable(alpha) for alpha in alphas]

            outputs_dst_transfer = []
            for src_i in range(len(source_loaders)):
                _, cur_hidden = encoders[src_i](batch1, batch2)
                cur_output = classifiers[src_i](cur_hidden)
                outputs_dst_transfer.append(cur_output)

            # outputs = [F.softmax(classifier(hidden), dim=1) for classifier in classifiers]
            outputs = [F.softmax(out, dim=1) for out in outputs_dst_transfer]

            alpha_cat = torch.zeros(size=(alphas[0].shape[0], len(encoders)))
            for col, a_list in enumerate(alphas):
                alpha_cat[:, col] = a_list

            cur_alpha_weights_stack = np.concatenate((cur_alpha_weights_stack, alpha_cat.detach().numpy()))

            output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                          for (alpha, output_i) in zip(alphas, outputs)])
            pred = output.data.max(dim=1)[1]
            # oracle_eq = compute_oracle(outputs, label, args)

            loss_batch = F.nll_loss(torch.log(output), label)
            loss += bs * loss_batch.item()

            # if args.eval_only:
            #     for i in range(batch.shape[0]):
            #         for j in range(len(alphas)):
            #             say("{:.4f}: [{:.4f}, {:.4f}], ".format(
            #                 alphas[j].data[i], outputs[j].data[i][0], outputs[j].data[i][1])
            #             )
            #         oracle_TF = "T" if oracle_eq[i] == 1 else colored("F", 'red')
            #         say("gold: {}, pred: {}, oracle: {}\n".format(label[i], pred[i], oracle_TF))
            #     say("\n")
                # print torch.cat(
                #         [
                #             torch.cat([ x.unsqueeze(1) for x in alphas ], 1),
                #             torch.cat([ x for x in outputs ], 1)
                #         ], 1
                #     )

            y_true += label.tolist()
            y_pred += pred.tolist()
            correct += pred.eq(label).sum()
            # oracle_correct += oracle_eq.sum()
            tot_cnt += output.size(0)
            y_score += output[:, 1].data.tolist()
    elif args.base_model == "rnn":
        for batch1, batch2, batch3, batch4, label in valid_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                batch3 = batch3.cuda()
                batch4 = batch4.cuda()
                label = label.cuda()

            # batch1 = Variable(batch1)
            # batch2 = Variable(batch2)
            bs = len(batch1)

            # hidden = encoder(batch)

            _, hidden_dst = encoder_dst(batch1, batch2, batch3, batch4)

            # source_ids = range(len(domain_encs))
            if args.metric == "biaffine":
                alphas = [biaffine_metric_fast(hidden_dst, mu[0], Us[0]) \
                          for mu in domain_encs]
            else:
                alphas = [mahalanobis_metric_fast(hidden_dst, mu[0], U, mu[1], P, mu[2], N) \
                          for (mu, U, P, N) in zip(domain_encs, Us, Ps, Ns)]
            # alphas = [ (1 - x / sum(alphas)) for x in alphas ]
            alphas = softmax(alphas)
            if args.cuda:
                alphas = [alpha.cuda() for alpha in alphas]
            alphas = [Variable(alpha) for alpha in alphas]

            outputs_dst_transfer = []
            for src_i in range(len(source_loaders)):
                _, cur_hidden = encoders[src_i](batch1, batch2, batch3, batch4)
                cur_output = classifiers[src_i](cur_hidden)
                outputs_dst_transfer.append(cur_output)

            # outputs = [F.softmax(classifier(hidden), dim=1) for classifier in classifiers]
            outputs = [F.softmax(out, dim=1) for out in outputs_dst_transfer]

            alpha_cat = torch.zeros(size=(alphas[0].shape[0], len(encoders)))
            for col, a_list in enumerate(alphas):
                alpha_cat[:, col] = a_list

            cur_alpha_weights_stack = np.concatenate((cur_alpha_weights_stack, alpha_cat.detach().numpy()))

            output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                          for (alpha, output_i) in zip(alphas, outputs)])
            pred = output.data.max(dim=1)[1]
            # oracle_eq = compute_oracle(outputs, label, args)

            loss_batch = F.nll_loss(torch.log(output), label)
            loss += bs * loss_batch.item()

            y_true += label.tolist()
            y_pred += pred.tolist()
            correct += pred.eq(label).sum()
            # oracle_correct += oracle_eq.sum()
            tot_cnt += output.size(0)
            y_score += output[:, 1].data.tolist()
    else:
        raise NotImplementedError

    alpha_weights = np.mean(cur_alpha_weights_stack, axis=0)
    print("alpha weights", alpha_weights)

    if thr is not None:
        print("using threshold %.4f" % thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1
    else:
        # print("y_score", y_score)
        pass

    loss /= tot_cnt

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    # print("y_score", y_score)
    auc = roc_auc_score(y_true, y_score)
    print("Loss: {:.4f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}".format(
        loss, auc * 100, prec * 100, rec * 100, f1 * 100))

    best_thr = None
    metric = [auc, prec, rec, f1]

    if return_best_thrs:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        print("best threshold={:.4f}, f1={:.4f}".format(best_thr, np.max(f1s)))

        writer.add_scalar('val_loss',
                          loss,
                          epoch)
    else:
        writer.add_scalar('test_f1',
                          f1,
                          epoch)

    acc = float(correct) / tot_cnt
    oracle_acc = float(oracle_correct) / tot_cnt
    # return (acc, oracle_acc), confusion_matrix(y_true, y_pred)
    return best_thr, metric


def predict(args):
    encoder, classifiers, Us, Ps, Ns = torch.load(args.load_model)
    map(lambda m: m.eval(), [encoder] + classifiers)

    # args = argparser.parse_args()
    # say(args)
    if args.cuda:
        map(lambda m: m.cuda(), [encoder] + classifiers)
        Us = [U.cuda() for U in Us]
        Ps = [P.cuda() for P in Ps]
        Ns = [N.cuda() for N in Ns]

    say("\nTransferring from %s to %s\n" % (args.train, args.test))
    source_train_sets = args.train.split(',')
    train_loaders = []
    for source in source_train_sets:
        filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (source))
        train_dataset = AmazonDataset(filepath)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        train_loaders.append(train_loader)

    test_filepath = os.path.join(DATA_DIR, "%s_test.svmlight" % (args.test))
    test_dataset = AmazonDataset(test_filepath)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    mats = [Us, Ps, Ns]
    (acc, oracle_acc), confusion_mat = evaluate(
        encoder, classifiers,
        mats,
        [train_loaders, test_loader],
        args
    )
    say(colored("Test accuracy/oracle {:.4f}/{:.4f}\n".format(acc, oracle_acc), 'red'))


def train(args):
    ''' Training Strategy
    Input: source = {S1, S2, ..., Sk}, target = {T}
    Train:
        Approach 1: fix metric and learn encoder only
        Approach 2: learn metric and encoder alternatively
    '''

    # test_mahalanobis_metric() and return

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    say('cuda is available %s\n' % args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed + args.seed_delta)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + args.seed_delta)

    source_train_sets = args.train.split(',')
    print("sources", source_train_sets)

    # encoder_class = get_model_class("mlp")
    # encoder_class.add_config(argparser)
    encoders_src = []
    for src_i in range(len(source_train_sets)):
        cur_model_dir = os.path.join(settings.OUT_DIR, source_train_sets[src_i])

        if args.base_model == "cnn":
            encoder_class = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                      mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                      mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                      mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                      mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
        elif args.base_model == "rnn":
            encoder_class = BiLSTM(vocab_size=args.max_vocab_size,
                   embedding_size=args.embedding_size,
                   hidden_size=args.hidden_size,
                   dropout=args.dropout)
        else:
            raise NotImplementedError
        if args.cuda:
            encoder_class.load_state_dict(torch.load(os.path.join(cur_model_dir, "{}-match-best-now.mdl".format(args.base_model))))
        else:
            encoder_class.load_state_dict(torch.load(os.path.join(cur_model_dir, "{}-match-best-now.mdl".format(args.base_model)), map_location=torch.device('cpu')))

        encoders_src.append(encoder_class)

    dst_pretrain_dir = os.path.join(settings.OUT_DIR, args.test)
    if args.base_model == "cnn":
        encoder_dst_pretrain = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                         mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                         mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                         mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                         mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    elif args.base_model == "rnn":
        encoder_dst_pretrain = BiLSTM(vocab_size=args.max_vocab_size,
                   embedding_size=args.embedding_size,
                   hidden_size=args.hidden_size,
                   dropout=args.dropout)
    else:
        raise NotImplementedError
    if args.cuda:
        encoder_dst_pretrain.load_state_dict(torch.load(os.path.join(dst_pretrain_dir, "{}-match-best-now.mdl".format(args.base_model))))
    else:
        encoder_dst_pretrain.load_state_dict(torch.load(os.path.join(dst_pretrain_dir, "{}-match-best-now.mdl".format(args.base_model)), map_location=torch.device('cpu')))

    critic_class = get_critic_class(args.critic)
    critic_class.add_config(argparser)

    args = argparser.parse_args()
    say(args)

    # encoder is shared across domains
    # encoder = encoder_class(args)

    say("Transferring from %s to %s\n" % (args.train, args.test))
    source_train_sets = args.train.split(',')
    train_loaders = []
    Us = []
    Ps = []
    Ns = []
    Ws = []
    Vs = []
    # Ms = []
    for source in source_train_sets:
        if args.base_model == "cnn":
            train_dataset = ProcessedCNNInputDataset(source, "train")
        elif args.base_model == "rnn":
            train_dataset = ProcessedRNNInputDataset(source, "train")
        else:
            raise NotImplementedError
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        train_loaders.append(train_loader)

        if args.metric == "biaffine":
            U = torch.FloatTensor(encoders_src[0].n_d, encoders_src[0].n_d)
            W = torch.FloatTensor(encoders_src[0].n_d, 1)
            nn.init.xavier_uniform_(W)
            Ws.append(W)
            V = torch.FloatTensor(encoders_src[0].n_d, 1)
            nn.init.xavier_uniform_(V)
            Vs.append(V)
        else:
            U = torch.FloatTensor(encoders_src[0].n_d, args.m_rank)

        nn.init.xavier_uniform_(U)
        Us.append(U)
        P = torch.FloatTensor(encoders_src[0].n_d, args.m_rank)
        nn.init.xavier_uniform_(P)
        Ps.append(P)
        N = torch.FloatTensor(encoders_src[0].n_d, args.m_rank)
        nn.init.xavier_uniform_(N)
        Ns.append(N)
        # Ms.append(U.mm(U.t()))

    # unl_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    # assert (os.path.exists(unl_filepath))
    # unl_dataset = AmazonDomainDataset(unl_filepath)
    # unl_loader = data.DataLoader(
    #     unl_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0
    # )

    if args.base_model == "cnn":
        train_dataset_dst = ProcessedCNNInputDataset(args.test, "train")
        valid_dataset = ProcessedCNNInputDataset(args.test, "valid")
        test_dataset = ProcessedCNNInputDataset(args.test, "test")

    elif args.base_model == "rnn":
        train_dataset_dst = ProcessedRNNInputDataset(args.test, "train")
        valid_dataset = ProcessedRNNInputDataset(args.test, "valid")
        test_dataset = ProcessedRNNInputDataset(args.test, "test")
    else:
        raise NotImplementedError
    train_loader_dst = data.DataLoader(
        train_dataset_dst,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    classifiers = []
    for source in source_train_sets:
        classifier = nn.Linear(encoders_src[0].n_out, 2)  # binary classification
        # nn.init.xavier_normal(classifier.weight)
        # nn.init.constant(classifier.bias, 0.1)
        classifiers.append(classifier)

    critic = critic_class(encoders_src[0], args)

    # if args.save_model:
    #     say(colored("Save model to {}\n".format(args.save_model + ".init"), 'red'))
    #     torch.save([encoder, classifiers, Us, Ps, Ns], args.save_model + ".init")

    if args.cuda:
        map(lambda m: m.cuda(), [critic, encoder_dst_pretrain] + classifiers + encoders_src)
        Us = [Variable(U.cuda(), requires_grad=True) for U in Us]
        Ps = [Variable(P.cuda(), requires_grad=True) for P in Ps]
        Ns = [Variable(N.cuda(), requires_grad=True) for N in Ns]
        if args.metric == "biaffine":
            Ws = [Variable(W.cuda(), requires_grad=True) for W in Ws]
            Vs = [Variable(V.cuda(), requires_grad=True) for V in Vs]

    # Ms = [ U.mm(U.t()) for U in Us ]

    # say("\nEncoder: {}\n".format(encoder))
    for i, classifier in enumerate(classifiers):
        say("Classifier-{}: {}\n".format(i, classifier))
    say("Critic: {}\n".format(critic))

    requires_grad = lambda x: x.requires_grad
    task_params = []
    # task_params = list(encoder.parameters())
    # for encoder in encoders_src:
    #     task_params += encoder.parameters()
    for classifier in classifiers:
        task_params += list(classifier.parameters())
    task_params += list(critic.parameters())
    task_params += Us
    task_params += Ps
    task_params += Ns
    if args.metric == "biaffine":
        task_params += Ws
        task_params += Vs

    if args.base_model == "cnn":
        optim_model = optim.Adagrad(
            filter(requires_grad, task_params),
            lr=args.lr,
            weight_decay=1e-4
        )
    elif args.base_model == "rnn":
        optim_model = optim.Adam(
            filter(requires_grad, task_params),
            lr=args.lr,
            weight_decay=1e-4
        )
    else:
        raise NotImplementedError

    say("Training will begin from scratch\n")

    best_dev = 0
    # best_test = 0
    iter_cnt = 0
    min_loss_val = None
    best_test_results = None

    model_dir = os.path.join(settings.OUT_DIR, args.test)

    for epoch in range(args.max_epoch):
        print("training epoch", epoch)
        if args.metric == "biaffine":
            mats = [Us, Ws, Vs]
        else:
            mats = [Us, Ps, Ns]

        iter_cnt = train_epoch(
            iter_cnt,
            [encoders_src, encoder_dst_pretrain], classifiers, critic,
            mats,
            [train_loaders, train_loader_dst, valid_loader],
            args,
            optim_model,
            epoch
        )

        thr, metrics_val = evaluate(
            epoch,
            [encoders_src, encoder_dst_pretrain], classifiers,
            mats,
            [train_loaders, valid_loader],
            True,
            args
        )
        # say("Dev accuracy/oracle: {:.4f}/{:.4f}\n".format(curr_dev, oracle_curr_dev))
        _, metrics_test = evaluate(
            epoch,
            [encoders_src, encoder_dst_pretrain], classifiers,
            mats,
            [train_loaders, test_loader],
            False,
            args,
            thr=thr
        )
        # say("Test accuracy/oracle: {:.4f}/{:.4f}\n".format(curr_test, oracle_curr_test))

        #TODO
        # if curr_dev >= best_dev:
        #     best_dev = curr_dev
        #     best_test = curr_test
        #     print(confusion_mat)
        #     if args.save_model:
        #         say(colored("Save model to {}\n".format(args.save_model + ".best"), 'red'))
        #         torch.save([encoder, classifiers, Us, Ps, Ns], args.save_model + ".best")
        if min_loss_val is None or min_loss_val > metrics_val[0]:
            min_loss_val = metrics_val[0]
            best_test_results = metrics_test
            # say(colored("Min valid"))
            torch.save([classifiers, Us, Ps, Ns],
                       os.path.join(model_dir, "{}_{}_moe_best_now.mdl".format(args.test, args.base_model)))
        say("\n")

    # say(colored("Best test accuracy {:.4f}\n".format(best_test), 'red'))
    say(colored("Min valid loss: {:.4f}, best test results, "
                "AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
        min_loss_val, best_test_results[1]*100, best_test_results[2]*100,
        best_test_results[3]*100, best_test_results[4]*100
    )))


def test_mahalanobis_metric():
    p = torch.FloatTensor(1, 5).normal_()
    S = torch.FloatTensor(4, 5).normal_()
    p = Variable(p)  # .cuda()
    S = Variable(S)  # .cuda()
    print(p, S)
    encoder = nn.Sequential(nn.Linear(5, 5), nn.ReLU())
    encoder = encoder  # .cuda()
    nn.init.xavier_normal(encoder[0].weight)
    nn.init.constant(encoder[0].bias, 0.1)
    print(encoder[0].weight)
    d = mahalanobis_metric(p, S, args, encoder)
    print(d)


def train_cnn_moe_stack(args):
    save_model_dir = os.path.join(settings.OUT_DIR, args.test)
    classifiers, Us, Ps, Ns = torch.load(os.path.join(save_model_dir, "{}_moe_best_now.mdl".format(args.test)))

    source_train_sets = args.train.split(',')

    encoders_src = []
    for src_i in range(len(source_train_sets)):
        cur_model_dir = os.path.join(settings.OUT_DIR, source_train_sets[src_i])

        encoder_class = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                      mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                      mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                      mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                      mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
        if args.cuda:
            encoder_class.load_state_dict(torch.load(os.path.join(cur_model_dir, "cnn-match-best-now.mdl")))
        else:
            encoder_class.load_state_dict(torch.load(os.path.join(cur_model_dir, "cnn-match-best-now.mdl"), map_location=torch.device('cpu')))

        encoders_src.append(encoder_class)

    dst_pretrain_dir = os.path.join(settings.OUT_DIR, args.test)
    encoder_dst_pretrain = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                         mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                         mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                         mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                         mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    if args.cuda:
        encoder_dst_pretrain.load_state_dict(torch.load(os.path.join(dst_pretrain_dir, "cnn-match-best-now.mdl")))
    else:
        encoder_dst_pretrain.load_state_dict(torch.load(os.path.join(dst_pretrain_dir, "cnn-match-best-now.mdl"), map_location=torch.device('cpu')))

    map(lambda m: m.eval(), encoders_src + classifiers + [encoder_dst_pretrain])

    if args.cuda:
        map(lambda m: m.cuda(), [encoder_dst_pretrain] + classifiers + encoders_src)
        Us = [U.cuda() for U in Us]
        Ps = [P.cuda() for P in Ps]
        Ns = [N.cuda() for N in Ns]

    train_loaders = []
    for source in source_train_sets:
        train_dataset = ProcessedCNNInputDataset(source, "train")
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        train_loaders.append(train_loader)
    train_dataset_dst = ProcessedCNNInputDataset(args.test, "train")
    train_loader_dst = data.DataLoader(
        train_dataset_dst,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    valid_dataset = ProcessedCNNInputDataset(args.test, "valid")
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_dataset = ProcessedCNNInputDataset(args.test, "test")
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    domain_encs = domain_encoding(train_loaders, args, encoders_src)

    meta_features = np.empty(shape=(0, 4))
    meta_labels = []

    for batch1, batch2, label in train_loader_dst:
        if args.cuda:
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
            label = label.cuda()

        _, hidden_dst = encoder_dst_pretrain(batch1, batch2)
        out_dst_cnn = encoder_dst_pretrain.fc_out(hidden_dst)
        # out_dst_cnn = torch.softmax(encoder_dst_pretrain.fc_out(hidden_dst), dim=1)

        if args.metric == "biaffine":
            alphas = [biaffine_metric_fast(hidden_dst, mu[0], Us[0]) \
                      for mu in domain_encs]
        else:
            alphas = [mahalanobis_metric_fast(hidden_dst, mu[0], U, mu[1], P, mu[2], N) \
                      for (mu, U, P, N) in zip(domain_encs, Us, Ps, Ns)]

        alphas = softmax(alphas)
        if args.cuda:
            alphas = [alpha.cuda() for alpha in alphas]
        alphas = [Variable(alpha) for alpha in alphas]

        outputs_dst_transfer = []
        for src_i in range(len(train_loaders)):
            _, cur_hidden = encoders_src[src_i](batch1, batch2)
            cur_output = classifiers[src_i](cur_hidden)
            outputs_dst_transfer.append(cur_output)

        # outputs = [F.softmax(classifier(hidden), dim=1) for classifier in classifiers]
        outputs = [F.softmax(out, dim=1) for out in outputs_dst_transfer]

        output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                      for (alpha, output_i) in zip(alphas, outputs)])

        if args.cuda:
            output = output.cpu()
            out_dst_cnn = out_dst_cnn.cpu()
            label = label.cpu()
        cur_feature = np.concatenate((out_dst_cnn.detach().numpy(), output.detach().numpy()), axis=1)
        meta_features = np.concatenate((meta_features, cur_feature), axis=0)
        meta_labels += label.data.tolist()

    print("meta features", meta_features)
    print("meta labels", meta_labels)

    meta_features_test = np.empty(shape=(0, 4))
    meta_labels_test = []

    for batch1, batch2, label in test_loader:
        if args.cuda:
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
            label = label.cuda()

        _, hidden_dst = encoder_dst_pretrain(batch1, batch2)
        out_dst_cnn = encoder_dst_pretrain.fc_out(hidden_dst)
        # out_dst_cnn = torch.softmax(encoder_dst_pretrain.fc_out(hidden_dst), dim=1)

        if args.metric == "biaffine":
            alphas = [biaffine_metric_fast(hidden_dst, mu[0], Us[0]) \
                      for mu in domain_encs]
        else:
            alphas = [mahalanobis_metric_fast(hidden_dst, mu[0], U, mu[1], P, mu[2], N) \
                      for (mu, U, P, N) in zip(domain_encs, Us, Ps, Ns)]

        alphas = softmax(alphas)
        if args.cuda:
            alphas = [alpha.cuda() for alpha in alphas]
        alphas = [Variable(alpha) for alpha in alphas]

        outputs_dst_transfer = []
        for src_i in range(len(train_loaders)):
            _, cur_hidden = encoders_src[src_i](batch1, batch2)
            cur_output = classifiers[src_i](cur_hidden)
            outputs_dst_transfer.append(cur_output)

        # outputs = [F.softmax(classifier(hidden), dim=1) for classifier in classifiers]
        outputs = [F.softmax(out, dim=1) for out in outputs_dst_transfer]

        output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                      for (alpha, output_i) in zip(alphas, outputs)])

        if args.cuda:
            output = output.cpu()
            out_dst_cnn = out_dst_cnn.cpu()
            label = label.cpu()
        cur_feature = np.concatenate((out_dst_cnn.detach().numpy(), output.detach().numpy()), axis=1)
        meta_features_test = np.concatenate((meta_features_test, cur_feature), axis=0)
        meta_labels_test += label.data.tolist()

    scaler = StandardScaler()
    meta_features_train_trans = scaler.fit_transform(meta_features)
    # clf = LogisticRegression(C=0.01, solver="saga").fit(meta_features_train_trans[:, 0:4], meta_labels)
    # print(clf.coef_)
    clf = SVC(kernel="linear", C=0.002, probability=True).fit(meta_features_train_trans[:, 0:4], meta_labels)
    # clf = RandomForestClassifier(max_depth=3, n_estimators=10).fit(meta_features_train_trans[:, 2:4], meta_labels)
    # clf = SVC(gamma=2, C=0.1, probability=True).fit(meta_features_train_trans[:, 0:4], meta_labels)
    # clf = LinearSVC().fit(meta_features_train_trans[:, 0:2], meta_labels)
    meta_features_test_trans = scaler.transform(meta_features_test)
    y_pred_test = clf.predict(meta_features_test_trans[:, 0:4])
    y_score_test = clf.predict_proba(meta_features_test_trans[:, 0:4])
    # print(y_score_test)

    prec, rec, f1, _ = precision_recall_fscore_support(meta_labels_test, y_pred_test, average="binary")
    # print("y_score", y_score)
    auc = roc_auc_score(meta_labels_test, y_score_test[:, 1])
    # auc = 0
    print("AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}".format(
        auc * 100, prec * 100, rec * 100, f1 * 100))


if __name__ == '__main__':
    train(args)
    # train_cnn_moe_stack(args)
