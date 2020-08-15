import sys, os, glob
import argparse
import time
import random
from copy import copy, deepcopy
from termcolor import colored, cprint
from copy import deepcopy
import numpy as np
import xgboost as xgb
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV

sys.path.append('../')
from msda_src.model_utils import get_model_class, get_critic_class
from msda_src.model_utils.domain_critic import ClassificationD, MMD, CoralD, WassersteinD
from msda_src.utils.io import AmazonDataset, AmazonDomainDataset
from msda_src.utils.io import say
from msda_src.utils.op import softmax

from dataset import ProcessedCNNInputDataset, ProcessedRNNInputDataset
from models.cnn import CNNMatchModel
from models.rnn import BiLSTM
from models.attn import MulInteractAttention

from utils import settings

import warnings

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser(description="Learning to Adapt from Multi-Source Domains")
argparser.add_argument("--cuda", action="store_true")
argparser.add_argument("--train", type=str, default="aff,author,paper,venue",
                       help="multi-source domains for training, separated with (,)")
argparser.add_argument("--test", type=str, default="aff",
                       help="target domain for testing")
argparser.add_argument("--eval_only", action="store_true")
argparser.add_argument("--critic", type=str, default="mmd")
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--batch_size_d", type=int, default=32)
argparser.add_argument("--max_epoch", type=int, default=200)
argparser.add_argument("--lr", type=float, default=5e-4)
argparser.add_argument("--lr_d", type=float, default=1e-4)
argparser.add_argument("--lambda_critic", type=float, default=0)
argparser.add_argument("--lambda_gp", type=float, default=0)
argparser.add_argument("--lambda_moe", type=float, default=1)
argparser.add_argument("--m_rank", type=int, default=10)
argparser.add_argument("--lambda_entropy", type=float, default=0.0)
argparser.add_argument("--load_model", type=str)
argparser.add_argument("--save_model", type=str)
argparser.add_argument("--base_model", type=str, default="cnn")
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
argparser.add_argument('--mat1-channel1', type=int, default=4, help='Matrix1 number of channels1.')
argparser.add_argument('--mat1-kernel-size1', type=int, default=3, help='Matrix1 kernel size1.')
argparser.add_argument('--mat1-channel2', type=int, default=8, help='Matrix1 number of channel2.')
argparser.add_argument('--mat1-kernel-size2', type=int, default=2, help='Matrix1 kernel size2.')
argparser.add_argument('--mat1-hidden', type=int, default=64, help='Matrix1 hidden dim.')
argparser.add_argument('--mat2-channel1', type=int, default=4, help='Matrix2 number of channels1.')
argparser.add_argument('--mat2-kernel-size1', type=int, default=2, help='Matrix2 kernel size1.')
argparser.add_argument('--mat2-hidden', type=int, default=64, help='Matrix2 hidden dim')
argparser.add_argument('--build-index-window', type=int, default=5, help='Matrix2 hidden dim')
argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
argparser.add_argument('--seed-delta', type=int, default=0, help='Random seed.')

argparser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (L2 loss on parameters).')
argparser.add_argument('--check-point', type=int, default=2, help="Check point")
argparser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")

args, _ = argparser.parse_known_args()

writer = SummaryWriter('runs/{}_sup_base_{}_simple_attn_moe_{}'.format(args.test, args.base_model, args.seed_delta))


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        # b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x * torch.log(x)
        b = -1.0 * b.sum()
        return b


def evaluate(epoch, encoders, classifiers, attn_mats, data_loader, return_best_thrs, args, thr=None):
    encoders, encoder_dst = encoders

    map(lambda m: m.eval(), encoders + classifiers + attn_mats)

    oracle_correct = 0
    correct = 0
    tot_cnt = 0
    y_true = []
    y_pred = []
    y_score = []
    loss = 0.

    n_sources = len(encoders)

    cur_alpha_weights_stack = np.empty(shape=(0, n_sources))
    if args.base_model == "cnn":
        for batch1, batch2, label in data_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                label = label.cuda()

            batch1 = Variable(batch1)
            batch2 = Variable(batch2)
            bs = len(batch1)

            _, hidden_from_dst_enc = encoder_dst(batch1, batch2)

            outputs_dst_transfer = []
            hidden_from_src_enc = []
            one_hot_sources = []

            for src_i in range(n_sources):
                _, cur_hidden = encoders[src_i](batch1, batch2)
                hidden_from_src_enc.append(cur_hidden)
                cur_output = classifiers[src_i](cur_hidden)
                outputs_dst_transfer.append(cur_output)
                cur_one_hot_sources = torch.zeros(size=(bs, n_sources))
                cur_one_hot_sources[:, src_i] = 1
                one_hot_sources.append(cur_one_hot_sources)

            source_ids = range(n_sources)
            support_ids = [x for x in source_ids]  # experts

            # source_alphas = [attn_mats[j](hidden_from_src_enc[j]).squeeze() for j in source_ids]
            source_alphas = [attn_mats[j](one_hot_sources[j]).squeeze() for j in source_ids]

            # source_alphas = [
            #     torch.bmm(attn_mats[j](hidden_from_src_enc[j]).unsqueeze(1), hidden_from_dst_enc.unsqueeze(2)).squeeze()
            #     for j in source_ids]
            # source_alphas = [attn_mats[j](hidden_from_src_enc[j], hidden_from_dst_enc).squeeze() for j in source_ids]

            support_alphas = [source_alphas[x] for x in support_ids]
            support_alphas = softmax(support_alphas)
            source_alphas = softmax(source_alphas)  # [ 32, 32, 32 ]
            alphas = source_alphas
            if args.cuda:
                alphas = [alpha.cuda() for alpha in alphas]

            outputs = [F.softmax(out, dim=1) for out in outputs_dst_transfer]

            alpha_cat = torch.zeros(size=(alphas[0].shape[0], n_sources))
            for col, a_list in enumerate(alphas):
                alpha_cat[:, col] = a_list

            cur_alpha_weights_stack = np.concatenate((cur_alpha_weights_stack, alpha_cat.detach().numpy()))
            output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                          for (alpha, output_i) in zip(alphas, outputs)])
            pred = output.data.max(dim=1)[1]

            loss_batch = F.nll_loss(torch.log(output), label)
            loss += bs * loss_batch.item()

            y_true += label.tolist()
            y_pred += pred.tolist()
            correct += pred.eq(label).sum()
            tot_cnt += output.size(0)
            y_score += output[:, 1].data.tolist()

    elif args.base_model == "rnn":
        for batch1, batch2, batch3, batch4, label in data_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                batch3 = batch3.cuda()
                batch4 = batch4.cuda()
                label = label.cuda()

            bs = len(batch1)

            _, hidden_from_dst_enc = encoder_dst(batch1, batch2, batch3, batch4)

            outputs_dst_transfer = []
            hidden_from_src_enc = []
            one_hot_sources = []

            for src_i in range(n_sources):
                _, cur_hidden = encoders[src_i](batch1, batch2, batch3, batch4)
                hidden_from_src_enc.append(cur_hidden)
                cur_output = classifiers[src_i](cur_hidden)
                outputs_dst_transfer.append(cur_output)
                cur_one_hot_sources = torch.zeros(size=(bs, n_sources))
                cur_one_hot_sources[:, src_i] = 1
                one_hot_sources.append(cur_one_hot_sources)

            source_ids = range(n_sources)
            support_ids = [x for x in source_ids]  # experts

            # source_alphas = [attn_mats[j](hidden_from_src_enc[j]).squeeze() for j in source_ids]
            # source_alphas = [
            #     torch.bmm(attn_mats[j](hidden_from_src_enc[j]).unsqueeze(1), hidden_from_dst_enc.unsqueeze(2)).squeeze()
            #     for j in source_ids]
            # source_alphas = [attn_mats[j](hidden_from_src_enc[j], hidden_from_dst_enc).squeeze() for j in source_ids]
            source_alphas = [attn_mats[j](one_hot_sources[j]).squeeze() for j in source_ids]

            support_alphas = [source_alphas[x] for x in support_ids]
            support_alphas = softmax(support_alphas)
            source_alphas = softmax(source_alphas)  # [ 32, 32, 32 ]
            alphas = source_alphas
            if args.cuda:
                alphas = [alpha.cuda() for alpha in alphas]

            outputs = [F.softmax(out, dim=1) for out in outputs_dst_transfer]

            alpha_cat = torch.zeros(size=(alphas[0].shape[0], n_sources))
            for col, a_list in enumerate(alphas):
                alpha_cat[:, col] = a_list

            cur_alpha_weights_stack = np.concatenate((cur_alpha_weights_stack, alpha_cat.detach().numpy()))
            output = sum([alpha.unsqueeze(1).repeat(1, 2) * output_i \
                          for (alpha, output_i) in zip(alphas, outputs)])
            pred = output.data.max(dim=1)[1]

            loss_batch = F.nll_loss(torch.log(output), label)
            loss += bs * loss_batch.item()

            y_true += label.tolist()
            y_pred += pred.tolist()
            correct += pred.eq(label).sum()
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
        pass

    loss /= tot_cnt

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    print("Loss: {:.4f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}".format(
        loss, auc * 100, prec * 100, rec * 100, f1 * 100))

    best_thr = None
    metric = [loss, auc, prec, rec, f1]

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

    return best_thr, metric


def train_epoch(iter_cnt, encoders, classifiers, attn_mats, train_loader_dst, args, optim_model, epoch):

    encoders, encoder_dst = encoders

    map(lambda m: m.train(), classifiers + encoders + attn_mats)

    moe_criterion = nn.NLLLoss()  # with log_softmax separated
    entropy_criterion = HLoss()

    loss_total = 0
    n_batch = 0
    n_sources = len(encoders)

    for batch in train_loader_dst:
        if args.base_model == "cnn":
            batch1, batch2, label = batch
        elif args.base_model == "rnn":
            batch1, batch2, batch3, batch4, label = batch
        else:
            raise NotImplementedError

        bs = len(label)

        iter_cnt += 1
        n_batch += 1
        if args.cuda:
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
            label = label.cuda()
            if args.base_model == "rnn":
                batch3 = batch3.cuda()
                batch4 = batch4.cuda()

        if args.base_model == "cnn":
            _, hidden_from_dst_enc = encoder_dst(batch1, batch2)
        elif args.base_model == "rnn":
            _, hidden_from_dst_enc = encoder_dst(batch1, batch2, batch3, batch4)
        else:
            raise NotImplementedError

        outputs_dst_transfer = []
        hidden_from_src_enc = []
        one_hot_sources = []
        for src_i in range(n_sources):
            if args.base_model == "cnn":
                _, cur_hidden = encoders[src_i](batch1, batch2)
                hidden_from_src_enc.append(cur_hidden)
            elif args.base_model == "rnn":
                _, cur_hidden = encoders[src_i](batch1, batch2, batch3, batch4)
                hidden_from_src_enc.append(cur_hidden)
            else:
                raise NotImplementedError
            cur_output = classifiers[src_i](cur_hidden)
            outputs_dst_transfer.append(cur_output)
            cur_one_hot_sources = torch.zeros(size=(bs, n_sources))
            cur_one_hot_sources[:, src_i] = 1
            one_hot_sources.append(cur_one_hot_sources)
        # print("one hot sources", one_hot_sources)

        optim_model.zero_grad()

        source_ids = range(n_sources)
        support_ids = [x for x in source_ids]  # experts
        # print("attn mats", attn_mats)
        # source_alphas = [attn_mats[j](hidden_from_src_enc[j]).squeeze() for j in source_ids]
        source_alphas = [attn_mats[j](one_hot_sources[j]).squeeze() for j in source_ids]

        # source_alphas = [attn_mats[j](hidden_from_src_enc[j], hidden_from_dst_enc).squeeze() for j in source_ids]
        # source_alphas = [torch.bmm(attn_mats[j](hidden_from_src_enc[j]).unsqueeze(1), hidden_from_dst_enc.unsqueeze(2)).squeeze() for j in source_ids]

        # print("source alphas", source_alphas[0].size(), source_alphas)

        support_alphas = [source_alphas[x] for x in support_ids]
        support_alphas = softmax(support_alphas)
        source_alphas = softmax(source_alphas)  # [ 32, 32, 32 ]

        if args.cuda:
            source_alphas = [alpha.cuda() for alpha in source_alphas]
        source_alphas = torch.stack(source_alphas, dim=0)
        source_alphas = source_alphas.permute(1, 0)

        loss_entropy = entropy_criterion(source_alphas)

        output_moe = sum([alpha.unsqueeze(1).repeat(1, 2) * F.softmax(outputs_dst_transfer[id], dim=1) \
                            for alpha, id in zip(support_alphas, support_ids)])
        loss_moe = moe_criterion(torch.log(output_moe), label)
        lambda_moe = args.lambda_moe
        loss = lambda_moe * loss_moe
        loss += args.lambda_entropy * loss_entropy
        loss_total += loss.item()
        loss.backward()
        optim_model.step()

        if iter_cnt % 5 == 0:
            say("{} MOE loss: {:.4f}, Entropy loss: {:.4f}, "
                "loss: {:.4f}\n"
                .format(iter_cnt,
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


def train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    say('cuda is available %s\n' % args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed + args.seed_delta)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + args.seed_delta)

    source_train_sets = args.train.split(',')
    print("sources", source_train_sets)

    pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))

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
            encoder_class = BiLSTM(pretrain_emb=pretrain_emb,
                                   vocab_size=args.max_vocab_size,
                                   embedding_size=args.embedding_size,
                                   hidden_size=args.hidden_size,
                                   dropout=args.dropout)
        else:
            raise NotImplementedError
        if args.cuda:
            encoder_class.load_state_dict(
                torch.load(os.path.join(cur_model_dir, "{}-match-best-now.mdl".format(args.base_model))))
        else:
            encoder_class.load_state_dict(
                torch.load(os.path.join(cur_model_dir, "{}-match-best-now.mdl".format(args.base_model)),
                           map_location=torch.device('cpu')))

        encoders_src.append(encoder_class)

    dst_pretrain_dir = os.path.join(settings.OUT_DIR, args.test)
    if args.base_model == "cnn":
        encoder_dst_pretrain = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                             mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                             mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                             mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                             mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    elif args.base_model == "rnn":
        encoder_dst_pretrain = BiLSTM(pretrain_emb=pretrain_emb,
                                      vocab_size=args.max_vocab_size,
                                      embedding_size=args.embedding_size,
                                      hidden_size=args.hidden_size,
                                      dropout=args.dropout)
    else:
        raise NotImplementedError

    args = argparser.parse_args()
    say(args)
    print()

    say("Transferring from %s to %s\n" % (args.train, args.test))

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
    attn_mats = []
    for source in source_train_sets:

        classifier = nn.Sequential(
            nn.Linear(encoders_src[0].n_out, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        # cur_att_weight = nn.Linear(len(encoders_src), 1, bias=True)
        cur_att_weight = nn.Linear(len(encoders_src), 1, bias=False)
        # nn.init.uniform_(cur_att_weight.weight)
        # print(cur_att_weight)
        cur_att_weight.weight = nn.Parameter(torch.ones(size=(1, len(encoders_src))), requires_grad=True)
        print("init cur att weight", cur_att_weight.weight)
        attn_mats.append(
            # nn.Linear(encoders_src[0].n_out, 1)
            cur_att_weight
            # nn.Linear(encoders_src[0].n_out, encoders_src[0].n_out)
            # MulInteractAttention(encoders_src[0].n_out, 16)
        )
        classifiers.append(classifier)
    print("classifier build", classifiers[0])

    if args.cuda:
        map(lambda m: m.cuda(), classifiers + encoders_src + attn_mats)

    for i, classifier in enumerate(classifiers):
        say("Classifier-{}: {}\n".format(i, classifier))

    requires_grad = lambda x: x.requires_grad
    task_params = []
    for src_i in range(len(classifiers)):
        task_params += list(classifiers[src_i].parameters())
        task_params += list(attn_mats[src_i].parameters())

    if args.base_model == "cnn":
        optim_model = optim.Adagrad(
            filter(requires_grad, task_params),
            lr=args.lr,
            weight_decay=1e-4  #TODO
        )
    elif args.base_model == "rnn":
        optim_model = optim.Adam(
            filter(requires_grad, task_params),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    say("Training will begin from scratch\n")

    iter_cnt = 0
    min_loss_val = None
    best_test_results = None
    model_dir = os.path.join(settings.OUT_DIR, args.test)

    for epoch in range(args.max_epoch):
        print("training epoch", epoch)

        iter_cnt = train_epoch(
            iter_cnt,
            [encoders_src, encoder_dst_pretrain], classifiers, attn_mats,
            train_loader_dst,
            args,
            optim_model,
            epoch
        )

        thr, metrics_val = evaluate(
            epoch,
            [encoders_src, encoder_dst_pretrain], classifiers,
            attn_mats,
            valid_loader,
            True,
            args
        )

        _, metrics_test = evaluate(
            epoch,
            [encoders_src, encoder_dst_pretrain], classifiers,
            attn_mats,
            test_loader,
            False,
            args,
            thr=thr
        )

        if min_loss_val is None or min_loss_val > metrics_val[0]:
            print("change val loss from {} to {}".format(min_loss_val, metrics_val[0]))
            min_loss_val = metrics_val[0]
            best_test_results = metrics_test
            torch.save([classifiers, attn_mats],
                       os.path.join(model_dir, "{}_{}_moe_simple_attn_best_now.mdl".format(args.test, args.base_model)))
        say("\n")
        writer.flush()

    say(colored("Min valid loss: {:.4f}, best test results, "
                "AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
        min_loss_val, best_test_results[1] * 100, best_test_results[2] * 100,
                      best_test_results[3] * 100, best_test_results[4] * 100
    )))


def train_moe_deep_stack(args):
    save_model_dir = os.path.join(settings.OUT_DIR, args.test)
    classifiers, attn_mats = torch.load(
        os.path.join(save_model_dir, "{}_{}_moe_best_now.mdl".format(args.test, args.base_model)))
    print("base model", args.base_model)
    print("classifier", classifiers[0])

    source_train_sets = args.train.split(',')
    pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))

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
            encoder_class = BiLSTM(pretrain_emb=pretrain_emb,
                                   vocab_size=args.max_vocab_size,
                                   embedding_size=args.embedding_size,
                                   hidden_size=args.hidden_size,
                                   dropout=args.dropout)
        else:
            raise NotImplementedError
        if args.cuda:
            encoder_class.load_state_dict(
                torch.load(os.path.join(cur_model_dir, "{}-match-best-now.mdl".format(args.base_model))))
        else:
            encoder_class.load_state_dict(
                torch.load(os.path.join(cur_model_dir, "{}-match-best-now.mdl".format(args.base_model)),
                           map_location=torch.device('cpu')))

        encoders_src.append(encoder_class)

    map(lambda m: m.eval(), encoders_src + classifiers + attn_mats)

    if args.cuda:
        map(lambda m: m.cuda(), classifiers + encoders_src + attn_mats)

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

    meta_features = np.empty(shape=(0, 192 + 2 * 8))
    meta_labels = []
    n_sources = len(encoders_src)
    encoders = encoders_src

    if args.base_model == "cnn":
        for batch1, batch2, label in train_loader_dst:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                label = label.cuda()

            outputs_dst_transfer = []
            hidden_from_src_enc = []
            for src_i in range(n_sources):
                _, cur_hidden = encoders[src_i](batch1, batch2)
                hidden_from_src_enc.append(cur_hidden)
                cur_output = classifiers[src_i](cur_hidden)
                outputs_dst_transfer.append(cur_output)

            source_ids = range(n_sources)
            support_ids = [x for x in source_ids]  # experts

            source_alphas = [attn_mats[j](hidden_from_src_enc[j]).squeeze() for j in source_ids]

            support_alphas = [source_alphas[x] for x in support_ids]
            support_alphas = softmax(support_alphas)
            source_alphas = softmax(source_alphas)  # [ 32, 32, 32 ]
            alphas = source_alphas



if __name__ == "__main__":
    train(args)
    writer.close()
