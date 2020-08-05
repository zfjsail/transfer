import sys, os, glob
import argparse
import time
import random
from copy import copy, deepcopy
from termcolor import colored, cprint

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from msda_src.model_utils import get_model_class, get_critic_class
from msda_src.model_utils.domain_critic import ClassificationD, MMD, CoralD, WassersteinD
from msda_src.utils.io import AmazonDataset, AmazonDomainDataset
from msda_src.utils.io import say
from msda_src.utils.op import softmax

from dataset import ProcessedCNNInputDataset, OAGDomainDataset
from models.cnn import CNNMatchModel

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from utils import settings

import warnings
warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser(description="Learning to Adapt from Multi-Source Domains")
argparser.add_argument("--cuda", action="store_true")
argparser.add_argument("--train", type=str, default="author,paper,aff",
                       help="multi-source domains for training, separated with (,)")
argparser.add_argument("--test", type=str, default="venue",
                       help="target domain for testing")
argparser.add_argument("--eval_only", action="store_true")
argparser.add_argument("--critic", type=str, default="mmd")
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--batch_size_d", type=int, default=32)
argparser.add_argument("--max_epoch", type=int, default=500)
argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--lr_d", type=float, default=1e-4)
argparser.add_argument("--lambda_critic", type=float, default=0)
argparser.add_argument("--lambda_gp", type=float, default=0)
argparser.add_argument("--lambda_moe", type=float, default=0)
argparser.add_argument("--lambda_mtl", type=float, default=0.3)
argparser.add_argument("--lambda_all", type=float, default=1)
argparser.add_argument("--lambda_dst", type=float, default=1)
argparser.add_argument("--m_rank", type=int, default=10)
argparser.add_argument("--lambda_entropy", type=float, default=0.0)
argparser.add_argument("--load_model", type=str)
argparser.add_argument("--save_model", type=str)
argparser.add_argument("--metric", type=str, default="biaffine",
                       help="mahalanobis: mahalanobis distance; biaffine: biaffine distance")

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

argparser.add_argument('--initial-accumulator-value', type=float, default=0.01, help='Initial accumulator value.')
argparser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (L2 loss on parameters).')
# argparser.add_argument('--dropout', type=float, default=0.2,
#                     help='Dropout rate (1 - keep probability).')
argparser.add_argument('--attn-dropout', type=float, default=0.,
                       help='Dropout rate (1 - keep probability).')
argparser.add_argument('--check-point', type=int, default=2, help="Check point")
argparser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")

args, _ = argparser.parse_known_args()

writer = SummaryWriter('runs/{}_mix_moe_{}'.format(args.test, args.seed_delta))


class WeightScaler(nn.Module):
    def __init__(self):
        super(WeightScaler, self).__init__()
        self.multp = nn.Parameter(torch.rand(1)) # requires_grad is True by default for Parameter


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
        for batch1, batch2, label in loader:
            if args.cuda:
                batch1 = Variable(batch1.cuda())
                batch2 = Variable(batch2.cuda())
            _, s_out = encoders[load_i](batch1, batch2)
            # print("s_out", s_out)
            S.append(s_out)
            if ind == 0:
                labels = label
            else:
                labels = torch.cat((labels, label), dim=0)
            ind += 1

        S = torch.cat(S, 0)

        # print("S", S)

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

        # print("mu_s", mu_S)
        # print("pos_mu_s", pos_mu_S)
        # print("neg_mu_s", neg_mu_S)

        statistics.append((mu_S, pos_mu_S, neg_mu_S))

    return statistics


TEMPERATURE = 4


def mahalanobis_metric_fast(p, mu, U, pos_mu, pos_U, neg_mu, neg_U):
    # covi = (cov + I).inverse()
    # print("p", type(p), p)
    # print("p", p.shape, p)
    # print("mu", mu.shape, mu)
    #
    # print("p - mu", p - mu)
    # print("U", U)
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
    classifiers, classifier_dst, classifier_mix = classifiers
    map(lambda m: m.train(), encoders + [encoder_dst, classifier_dst, critic, classifier_mix] + classifiers)

    train_loaders, train_loader_dst, unl_loader, valid_loader = data_loaders
    dup_train_loaders = deepcopy(train_loaders)

    # mtl_criterion = nn.CrossEntropyLoss()
    mtl_criterion = nn.NLLLoss()
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
    total = 0

    for batches, batches_dst, unl_batch in zip(zip(*train_loaders), train_loader_dst, unl_loader):
        train_batches1, train_batches2, train_labels = zip(*batches)
        # print("train batches1", train_labels[0].size())
        # print("train batches2", train_batches2)
        # print("train labels", train_labels)
        unl_critic_batch1, unl_critic_batch2, unl_critic_label = unl_batch
        # print("unl", unl_critic_batch1)
        batches1_dst, batches2_dst, labels_dst = batches_dst
        # print("batches1_dst", batches1_dst)
        # print("batches2_dst", batches2_dst)

        total += len(batches1_dst)

        iter_cnt += 1
        if args.cuda:
            train_batches1 = [batch.cuda() for batch in train_batches1]
            train_batches2 = [batch.cuda() for batch in train_batches2]
            train_labels = [label.cuda() for label in train_labels]

            batches1_dst = batches1_dst.cuda()
            batches2_dst = batches2_dst.cuda()
            labels_dst = labels_dst.cuda()

            unl_critic_batch1 = unl_critic_batch1.cuda()
            unl_critic_batch2 = unl_critic_batch2.cuda()
            unl_critic_label = unl_critic_label.cuda()

        # train_batches1 = [Variable(batch) for batch in train_batches1]
        # train_batches2 = [Variable(batch) for batch in train_batches2]
        # train_labels = [Variable(label) for label in train_labels]
        # unl_critic_batch1 = Variable(unl_critic_batch1)
        # unl_critic_batch2 = Variable(unl_critic_batch2)
        # unl_critic_label = Variable(unl_critic_label)

        optim_model.zero_grad()
        loss_train_dst = []
        loss_mtl = []
        loss_moe = []
        loss_kl = []
        loss_entropy = []
        loss_dan = []
        loss_all = []

        ms_outputs = []  # (n_sources, n_classifiers)
        hiddens = []
        hidden_corresponding_labels = []
        # labels = []

        _, hidden_dst = encoder_dst(batches1_dst, batches2_dst)
        cur_output_dst = classifier_dst(hidden_dst)
        cur_output_dst_mem = torch.softmax(cur_output_dst, dim=1)
        cur_output_dst = torch.log(cur_output_dst_mem)
        loss_train_dst.append(mtl_criterion(cur_output_dst, labels_dst))

        outputs_dst_transfer = []
        for i in range(len(train_batches1)):
            _, cur_hidden = encoders[i](batches1_dst, batches2_dst)
            cur_output = classifiers[i](cur_hidden)
            outputs_dst_transfer.append(cur_output)

        for i, (batch1, batch2, label) in enumerate(zip(train_batches1, train_batches2, train_labels)):  # source i
            _, hidden = encoders[i](batch1, batch2)
            outputs = []
            # create output matrix:
            #     - (i, j) indicates the output of i'th source batch using j'th classifier
            # print("hidden", hidden)
            # raise
            hiddens.append(hidden)
            for classifier in classifiers:
                output = classifier(hidden)
                output = torch.log_softmax(output, dim=1)
                # print("output", output)
                outputs.append(output)
            ms_outputs.append(outputs)
            hidden_corresponding_labels.append(label)
            # multi-task loss
            # print("ms & label", ms_outputs[i][i], label)
            loss_mtl.append(mtl_criterion(ms_outputs[i][i], label))
            # labels.append(label)

            if args.lambda_critic > 0:
                # critic_batch = torch.cat([batch, unl_critic_batch])
                critic_label = torch.cat([1 - unl_critic_label, unl_critic_label])
                # critic_label = torch.cat([1 - unl_critic_label] * len(train_batches) + [unl_critic_label])

                if isinstance(critic, ClassificationD):
                    critic_output = critic(torch.cat(hidden, encoders[i](unl_critic_batch1, unl_critic_batch2)))
                    loss_dan.append(critic.compute_loss(critic_output, critic_label))
                else:
                    critic_output = critic(hidden, encoders[i](unl_critic_batch1, unl_critic_batch2))
                    loss_dan.append(critic_output)

                    # critic_output = critic(torch.cat(hiddens), encoder(unl_critic_batch))
                    # loss_dan = critic_output
            else:
                loss_dan = Variable(torch.FloatTensor([0]))


        # assert (len(outputs) == len(outputs[0]))
        source_ids = range(len(train_batches1))
        # for i in source_ids:

        # support_ids = [x for x in source_ids if x != i]  # experts
        support_ids = [x for x in source_ids]  # experts

        # i = 0

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
            source_alphas = [metric(hidden_dst,  # i^th source
                                    hiddens[j].detach(),
                                    hidden_corresponding_labels[j],
                                    Us[j], Ps[j], Ns[j],
                                    args) for j in source_ids]

        support_alphas = [source_alphas[x] for x in support_ids]

        # print torch.cat([ x.unsqueeze(1) for x in support_alphas ], 1)
        support_alphas = softmax(support_alphas)

        # print("support_alphas after softmax", support_alphas)

        # meta-supervision: KL loss over \alpha and real source
        source_alphas = softmax(source_alphas)  # [ 32, 32, 32 ]
        source_labels = [torch.FloatTensor([x == len(train_batches1)]) for x in source_ids]  # one-hot
        if args.cuda:
            source_alphas = [alpha.cuda() for alpha in source_alphas]
            source_labels = [label.cuda() for label in source_labels]

        source_labels = Variable(torch.stack(source_labels, dim=0))  # 3*1
        # print("source labels", source_labels)
        source_alphas = torch.stack(source_alphas, dim=0)
        # print("source_alpha after stack", source_alphas)

        source_labels = source_labels.expand_as(source_alphas).permute(1, 0)
        source_alphas = source_alphas.permute(1, 0)
        loss_kl.append(kl_criterion(source_alphas, source_labels))

        # entropy loss over \alpha
        # entropy_loss = entropy_criterion(torch.stack(support_alphas, dim=0).permute(1, 0))
        # print source_alphas
        loss_entropy.append(entropy_criterion(source_alphas))

        output_moe_i = sum([alpha.unsqueeze(1).repeat(1, 2) * F.softmax(outputs_dst_transfer[id], dim=1) \
                            for alpha, id in zip(support_alphas, support_ids)])
        # output_moe_full = sum([ alpha.unsqueeze(1).repeat(1, 2) * F.softmax(ms_outputs[i][id], dim=1) \
        #                         for alpha, id in zip(full_alphas, source_ids) ])

        # print("output_moe_i & labels", output_moe_i, train_labels[i])
        loss_moe.append(moe_criterion(torch.log(output_moe_i), labels_dst))
        # loss_moe.append(moe_criterion(torch.log(output_moe_full), train_labels[i]))

        # print("labels_dst", labels_dst)

        # upper_out = classifier_mix(torch.cat((cur_output_dst_mem, output_moe_i), dim=1))
        upper_out = cur_output_dst_mem + classifier_mix.multp * output_moe_i
        loss_all = mtl_criterion(torch.log_softmax(upper_out, dim=1), labels_dst)

        loss_train_dst = sum(loss_train_dst)

        loss_mtl = sum(loss_mtl)
        # print("loss mtl", loss_mtl)
        # loss_mtl = loss_mtl.mean()
        loss_mtl /= len(source_ids)
        loss_moe = sum(loss_moe)
        # if iter_cnt < 400:
        #     lambda_moe = 0
        #     lambda_entropy = 0
        # else:
        lambda_moe = args.lambda_moe
        lambda_entropy = args.lambda_entropy
        # loss = (1 - lambda_moe) * loss_mtl + lambda_moe * loss_moe
        loss = args.lambda_mtl * loss_mtl + lambda_moe * loss_moe
        loss_kl = sum(loss_kl)
        loss_entropy = sum(loss_entropy)
        loss += args.lambda_entropy * loss_entropy
        loss += loss_train_dst * args.lambda_dst
        loss += loss_all * args.lambda_all

        loss_total += loss

        if args.lambda_critic > 0:
            loss_dan = sum(loss_dan)
            loss += args.lambda_critic * loss_dan

        loss.backward()
        optim_model.step()

        # print("loss entropy", loss_entropy)

        # print("mats", [Us, Ps, Ns])
        # for paras in task_paras:
        #     print(paras)
        #     for name, param in paras:
        #         if param.requires_grad:
        #             print(name, param.data)

        # for name, param in encoder.named_parameters():
        #     if param.requires_grad:
        #         # print(name, param.data)
        #         print(name, param.grad)

        for cls_i, classifier in enumerate(classifiers):
            for name, param in classifier.named_parameters():
                # print(cls_i, name, param.grad)
                pass

        if iter_cnt % 5 == 0:
            # [(mu_i, covi_i), ...]
            # domain_encs = domain_encoding(dup_train_loaders, args, encoder)
            if args.metric == "biaffine":
                mats = [Us, Ws, Vs]
            else:
                mats = [Us, Ps, Ns]

            # evaluate(
            #             #     [encoders, encoder_dst],
            #             #     [classifiers, classifier_dst, classifier_mix],
            #             #     mats,
            #             #     [dup_train_loaders, valid_loader],
            #             #     True,
            #             #     args
            #             # )

            # say("\r" + " " * 50)
            # TODO: print train acc as well
            # print("loss dan", loss_dan)
            say("{} MTL loss: {:.4f}, MOE loss: {:.4f}, DAN loss: {:.4f}, "
                "loss: {:.4f}\n"
                # ", dev acc/oracle: {:.4f}/{:.4f}"
                .format(iter_cnt,
                        loss_mtl.item(),
                        loss_moe.item(),
                        loss_dan.item(),
                        loss.item(),
                        # curr_dev,
                        # oracle_curr_dev
                        ))

    writer.add_scalar('training_loss',
                      loss_total / total,
                      epoch)

    say("\n")
    return iter_cnt


def compute_oracle(outputs, label, args):
    ''' Compute the oracle accuracy given outputs from multiple classifiers
    '''
    # oracle = torch.ByteTensor([0] * label.shape[0])
    oracle = torch.BoolTensor([0] * label.shape[0])
    if args.cuda:
        oracle = oracle.cuda()
    for i, output in enumerate(outputs):
        pred = output.data.max(dim=1)[1]
        # print("pred", pred)
        # print("label", label)
        oracle |= pred.eq(label.byte())
    return oracle


def evaluate(epoch, encoders, classifiers, mats, loaders, return_best_thrs, args, thr=None):
    ''' Evaluate model using MOE
    '''

    encoders, encoder_dst = encoders
    classifiers, classifier_dst, classifier_mix = classifiers

    map(lambda m: m.eval(), encoders + classifiers + [encoder_dst, classifier_dst, classifier_mix])

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

    source_ids = range(len(domain_encs))

    for batch1, batch2, label in valid_loader:
        if args.cuda:
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
            label = label.cuda()
        # print("eval labels", label)

        batch1 = Variable(batch1)
        batch2 = Variable(batch2)
        bs = len(batch1)
        # print("bs", len(batch1))

        _, hidden_dst = encoder_dst(batch1, batch2)
        cur_output_dst = classifier_dst(hidden_dst)
        cur_output_dst_mem = torch.softmax(cur_output_dst, dim=1)
        # print("mem", cur_output_dst_mem)
        cur_output_dst = torch.log(cur_output_dst_mem)

        outputs_dst_transfer = []
        for src_i in range(len(source_loaders)):
            _, cur_hidden = encoders[src_i](batch1, batch2)
            cur_output = classifiers[src_i](cur_hidden)
            outputs_dst_transfer.append(cur_output)

        # _, hidden = encoders[0](batch1, batch2)
        # source_ids = range(len(domain_encs))
        if args.metric == "biaffine":
            alphas = [biaffine_metric_fast(hidden_dst, mu[0], Us[0]) \
                      for mu in domain_encs]
        else:
            alphas = [mahalanobis_metric_fast(hidden_dst, mu[0], U, mu[1], P, mu[2], N) \
                      for (mu, U, P, N) in zip(domain_encs, Us, Ps, Ns)]
        # # alphas = [ (1 - x / sum(alphas)) for x in alphas ]
        alphas = softmax(alphas)
        if args.cuda:
            alphas = [alpha.cuda() for alpha in alphas]
        alphas = [Variable(alpha) for alpha in alphas]
        #
        # outputs = [F.softmax(classifier(hidden), dim=1) for classifier in classifiers]
        output_moe = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                      for (alpha, output_i) in zip(alphas, outputs_dst_transfer)])
        # pred = output.data.max(dim=1)[1]
        # oracle_eq = compute_oracle(outputs, label, args)

        # outputs = classifier_mix(torch.cat((cur_output_dst_mem, output_moe), dim=1))
        outputs = cur_output_dst_mem + classifier_mix.multp * output_moe
        # print("weight mix", classifier_mix.multp)
        outputs_upper_logits = torch.log_softmax(outputs, dim=1)
        # outputs_upper_logits = torch.log(cur_output_dst_mem)
        # print("outputs_upper_logits", outputs_upper_logits)
        pred = outputs_upper_logits.data.max(dim=1)[1]
        # oracle_eq = compute_oracle(outputs_upper_logits, label, args)

        loss_batch = F.nll_loss(outputs_upper_logits, label)
        loss += bs * loss_batch.item()

        # if args.eval_only:
        #     for i in range(batch1.shape[0]):
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
        # print("output", output[:, 1].data.tolist())
        y_score += outputs_upper_logits[:, 1].data.tolist()
        # print("cur y score", y_score)

        correct += pred.eq(label).sum()
        # oracle_correct += oracle_eq.sum()
        tot_cnt += outputs_upper_logits.size(0)

    # print("y_true", y_true)
    # print("y_pred", y_pred)

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


def evaluate_cross(encoder, classifiers, mats, loaders, return_best_thrs, args, thr=None):
    ''' Evaluate model using MOE
    '''
    map(lambda m: m.eval(), [encoder] + classifiers)

    if args.metric == "biaffine":
        Us, Ws, Vs = mats
    else:
        Us, Ps, Ns = mats

    source_loaders, valid_loaders_src = loaders
    domain_encs = domain_encoding(source_loaders, args, encoder)

    source_ids = range(len(valid_loaders_src))

    thresholds = []
    metrics = []
    alphas_weights = np.zeros(shape=(4, 4))

    for src_i in range(len(valid_loaders_src)):
        valid_loader = valid_loaders_src[src_i]

        oracle_correct = 0
        correct = 0
        tot_cnt = 0
        y_true = []
        y_pred = []
        y_score = []

        # support_ids = [x for x in source_ids if x != src_i]  # experts
        support_ids = [x for x in source_ids]  # experts
        cur_domain_encs = [domain_encs[x] for x in support_ids]
        cur_Us = [Us[x] for x in support_ids]
        cur_Ps = [Ps[x] for x in support_ids]
        cur_Ns = [Ns[x] for x in support_ids]

        cur_alpha_weights = [[]] * 4
        cur_alpha_weights_stack = np.empty(shape=(0, len(support_ids)))

        for batch1, batch2, label in valid_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                label = label.cuda()
            # print("eval labels", label)

            batch1 = Variable(batch1)
            batch2 = Variable(batch2)
            _, hidden = encoder(batch1, batch2)
            # source_ids = range(len(domain_encs))
            if args.metric == "biaffine":
                alphas = [biaffine_metric_fast(hidden, mu[0], Us[0]) \
                          for mu in domain_encs]
            else:
                alphas = [mahalanobis_metric_fast(hidden, mu[0], U, mu[1], P, mu[2], N) \
                          for (mu, U, P, N) in zip(cur_domain_encs, cur_Us, cur_Ps, cur_Ns)]
            # alphas = [ (1 - x / sum(alphas)) for x in alphas ]
            alphas = softmax(alphas)
            # print("alphas", alphas[0].mean(), alphas[1].mean(), alphas[2].mean())
            # print("alphas", alphas)

            alphas = []
            for al_i in range(len(support_ids)):
                alphas.append(torch.zeros(size=(batch1.size()[0],)))
            alphas[src_i] = torch.ones(size=(batch1.size()[0],))

            alpha_cat = torch.zeros(size=(alphas[0].shape[0], len(support_ids)))
            for col, a_list in enumerate(alphas):
                alpha_cat[:, col] = a_list
            cur_alpha_weights_stack = np.concatenate((cur_alpha_weights_stack, alpha_cat.detach().numpy()))
            # for j, supp_id in enumerate(support_ids):
            # cur_alpha_weights[supp_id] += alphas[j].data.tolist()
            # cur_alpha_weights[supp_id].append(alphas[j].mean().item())
            if args.cuda:
                alphas = [alpha.cuda() for alpha in alphas]
            alphas = [Variable(alpha) for alpha in alphas]

            outputs = [F.softmax(classifiers[j](hidden), dim=1) for j in support_ids]
            output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                          for (alpha, output_i) in zip(alphas, outputs)])
            # print("pred output", output)
            pred = output.data.max(dim=1)[1]
            oracle_eq = compute_oracle(outputs, label, args)

            if args.eval_only:
                for i in range(batch1.shape[0]):
                    for j in range(len(alphas)):
                        say("{:.4f}: [{:.4f}, {:.4f}], ".format(
                            alphas[j].data[i], outputs[j].data[i][0], outputs[j].data[i][1])
                        )
                    oracle_TF = "T" if oracle_eq[i] == 1 else colored("F", 'red')
                    say("gold: {}, pred: {}, oracle: {}\n".format(label[i], pred[i], oracle_TF))
                say("\n")
                # print torch.cat(
                #         [
                #             torch.cat([ x.unsqueeze(1) for x in alphas ], 1),
                #             torch.cat([ x for x in outputs ], 1)
                #         ], 1
                #     )

            y_true += label.tolist()
            y_pred += pred.tolist()
            y_score += output[:, 1].data.tolist()
            correct += pred.eq(label).sum()
            oracle_correct += oracle_eq.sum()
            tot_cnt += output.size(0)

        # print("y_true", y_true)
        # print("y_pred", y_pred)

        # for j in support_ids:
        #     print(src_i, j, cur_alpha_weights[j])
        #     alphas_weights[src_i, j] = np.mean(cur_alpha_weights[j])
        # print(alphas_weights)
        alphas_weights[src_i, support_ids] = np.mean(cur_alpha_weights_stack, axis=0)

        if thr is not None:
            print("using threshold %.4f" % thr[src_i])
            y_score = np.array(y_score)
            y_pred = np.zeros_like(y_score)
            y_pred[y_score > thr[src_i]] = 1

        # prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

        acc = float(correct) / tot_cnt
        oracle_acc = float(oracle_correct) / tot_cnt
        # print("source", src_i, "validation results: precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
        #     prec*100, rec*100, f1*100))
        # return (acc, oracle_acc), confusion_matrix(y_true, y_pred)

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        auc = roc_auc_score(y_true, y_score)
        print("source {}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}".format(
            src_i, auc * 100, prec * 100, rec * 100, f1 * 100))

        metrics.append([auc, prec, rec, f1])

        if return_best_thrs:
            precs, recs, thrs = precision_recall_curve(y_true, y_score)
            f1s = 2 * precs * recs / (precs + recs)
            f1s = f1s[:-1]
            thrs = thrs[~np.isnan(f1s)]
            f1s = f1s[~np.isnan(f1s)]
            best_thr = thrs[np.argmax(f1s)]
            print("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
            thresholds.append(best_thr)

    print("source domain weight matrix\n", alphas_weights)

    metrics = np.array(metrics)
    return thresholds, metrics, alphas_weights


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

    encoders = []
    for _ in range(len(source_train_sets)):
        # encoder_class = get_model_class("mlp")
        encoder_class = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                      mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                      mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                      mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                      mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)

        # encoder_class.add_config(argparser)
        encoders.append(encoder_class)

    encoder_dst = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)

    critic_class = get_critic_class(args.critic)
    critic_class.add_config(argparser)

    args = argparser.parse_args()
    say(args)

    # encoder is shared across domains
    # encoder = encoder_class(args)
    # encoder = encoder_class

    print()
    print("encoder", encoders[0])

    say("Transferring from %s to %s\n" % (args.train, args.test))
    train_loaders = []
    # valid_loaders_src = []
    # test_loaders_src = []
    Us = []
    Ps = []
    Ns = []
    Ws = []
    Vs = []
    # Ms = []

    for source in source_train_sets:
        # filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (source))
        filepath = os.path.join(settings.DOM_ADAPT_DIR, "{}_train.pkl".format(source))
        assert (os.path.exists(filepath))
        # train_dataset = AmazonDataset(filepath)
        train_dataset = ProcessedCNNInputDataset(source, "train")
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        train_loaders.append(train_loader)

        # cur_valid_dataset = ProcessedCNNInputDataset(source, "valid")
        # cur_valid_loader = data.DataLoader(
        #     cur_valid_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=0
        # )
        # valid_loaders_src.append(cur_valid_loader)
        #
        # cur_test_dataset = ProcessedCNNInputDataset(source, "test")
        # cur_test_loader = data.DataLoader(
        #     cur_test_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=0
        # )
        # test_loaders_src.append(cur_test_loader)

        if args.metric == "biaffine":
            U = torch.FloatTensor(encoders[0].n_d, encoders[0].n_d)
            W = torch.FloatTensor(encoders[0].n_d, 1)
            nn.init.xavier_uniform(W)
            Ws.append(W)
            V = torch.FloatTensor(encoders[0].n_d, 1)
            nn.init.xavier_uniform(V)
            Vs.append(V)
        else:
            U = torch.FloatTensor(encoders[0].n_d, args.m_rank)

        nn.init.xavier_uniform_(U)
        Us.append(U)
        P = torch.FloatTensor(encoders[0].n_d, args.m_rank)
        nn.init.xavier_uniform_(P)
        Ps.append(P)
        N = torch.FloatTensor(encoders[0].n_d, args.m_rank)
        nn.init.xavier_uniform_(N)
        Ns.append(N)
        # Ms.append(U.mm(U.t()))

    # unl_filepath = os.path.join(DATA_DIR, "%s_train.svmlight" % (args.test))
    unl_filepath = os.path.join(settings.DOM_ADAPT_DIR, "{}_train.pkl".format(args.test))
    print("****************", unl_filepath)
    assert (os.path.exists(unl_filepath))
    # unl_dataset = AmazonDomainDataset(unl_filepath)  # using domain as labels
    unl_dataset = OAGDomainDataset(args.test, "train")
    unl_loader = data.DataLoader(
        unl_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    train_dataset_dst = ProcessedCNNInputDataset(args.test, "train")
    train_loader_dst = data.DataLoader(
        train_dataset_dst,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # valid_filepath = os.path.join(DATA_DIR, "%s_test.svmlight" % (args.test))  # No dev files
    # valid_dataset = AmazonDataset(valid_filepath)
    valid_dataset = ProcessedCNNInputDataset(args.test, "valid")
    print("valid y", len(valid_dataset), valid_dataset.y)
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # test_filepath = os.path.join(DATA_DIR, "%s_test.svmlight" % (args.test))
    # assert (os.path.exists(test_filepath))
    # test_dataset = AmazonDataset(test_filepath)
    test_dataset = ProcessedCNNInputDataset(args.test, "test")
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    say("Corpus loaded.\n")

    classifiers = []
    for source in source_train_sets:  # only one layer
        classifier = nn.Linear(encoders[0].n_out, 2)  # binary classification
        # classifier = encoder.fc_out
        # nn.init.xavier_normal(classifier.weight)
        # nn.init.constant(classifier.bias, 0.1)
        classifiers.append(classifier)

    classifier_dst = nn.Linear(encoder_dst.n_out, 2)
    # classifier_mix = nn.Linear(2, 2)
    classifier_mix = WeightScaler()

    critic = critic_class(encoders[0], args)

    # if args.save_model:
    #     say(colored("Save model to {}\n".format(args.save_model + ".init"), 'red'))
    #     torch.save([encoder, classifiers, Us, Ps, Ns], args.save_model + ".init")

    if args.cuda:
        map(lambda m: m.cuda(), [encoder_dst, critic, classifier_dst, classifier_mix] + encoders + classifiers)
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
    # task_params = list(encoder.parameters())
    task_params = []
    for encoder in encoders:
        task_params += encoder.parameters()
    task_params += encoder_dst.parameters()
    for classifier in classifiers:
        task_params += list(classifier.parameters())
    task_params += classifier_dst.parameters()
    task_params += classifier_mix.parameters()
    # task_params += [classifier_mix.data]
    task_params += list(critic.parameters())
    task_params += Us
    task_params += Ps
    task_params += Ns
    if args.metric == "biaffine":
        task_params += Ws
        task_params += Vs

    optim_model = optim.Adagrad(  # use adagrad instead of adam
        filter(requires_grad, task_params),
        lr=args.lr,
        weight_decay=1e-4
    )

    say("Training will begin from scratch\n")

    best_dev = 0
    best_test = 0
    iter_cnt = 0

    # encoder.load_state_dict(torch.load(os.path.join(settings.OUT_VENUE_DIR, "venue-matching-cnn.mdl")))

    for epoch in range(args.max_epoch):
        say("epoch: {}\n".format(epoch))
        if args.metric == "biaffine":
            mats = [Us, Ws, Vs]
        else:
            mats = [Us, Ps, Ns]

        iter_cnt = train_epoch(
            iter_cnt,
            [encoders, encoder_dst],
            [classifiers, classifier_dst, classifier_mix], critic,
            mats,
            [train_loaders, train_loader_dst, unl_loader, valid_loader],
            args,
            optim_model,
            epoch
        )

        # thrs, metrics_val, src_weights_val = evaluate_cross(
        #     encoder, classifiers,
        #     mats,
        #     [train_loaders, valid_loaders_src],
        #     return_best_thrs=True,
        #     args=args
        # )
        #
        # _, metrics_test, src_weights_test = evaluate_cross(
        #     encoder, classifiers,
        #     mats,
        #     [train_loaders, test_loaders_src],
        #     return_best_thrs=False,
        #     args=args,
        #     thr=thrs
        # )

        thr, metrics_val = evaluate(
            epoch,
            [encoders, encoder_dst],
            [classifiers, classifier_dst, classifier_mix],
            mats,
            [train_loaders, valid_loader],
            True,
            args
        )
        # say("Dev accuracy/oracle: {:.4f}/{:.4f}\n".format(curr_dev, oracle_curr_dev))
        _, metrics_test = evaluate(
            epoch,
            [encoders, encoder_dst],
            [classifiers, classifier_dst, classifier_mix],
            mats,
            [train_loaders, test_loader],
            False,
            args,
            thr=thr
        )
        # say("Test accuracy/oracle: {:.4f}/{:.4f}\n".format(curr_test, oracle_curr_test))

        # if curr_dev >= best_dev:
        #     best_dev = curr_dev
        #     best_test = curr_test
        #     print(confusion_mat)
        #     if args.save_model:
        #         say(colored("Save model to {}\n".format(args.save_model + ".best"), 'red'))
        #         torch.save([encoder, classifiers, Us, Ps, Ns], args.save_model + ".best")
        say("\n")

    say(colored("Best test accuracy {:.4f}\n".format(best_test), 'red'))


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


# import argparse

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    if args.cuda:
        torch.cuda.manual_seed(0)

    print("eval only", args.eval_only)
    if args.eval_only:
        predict(args)
    else:
        train(args)
    writer.close()
