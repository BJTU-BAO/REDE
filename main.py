import torch
import argparse
import logging
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCNSVD
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset
import datetime
import sys
sys.path.append("..")
from MultiGCN.model import DeepGCN
import sys
sys.path.append("..")
from MultiGCN.utils import process, compute_auc, compute_f1, largeDataprocess
from sklearn.preprocessing import MinMaxScaler, scale
from feature_filp.feature_flip_utils import feature_flip
import os.path as osp
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'git2', 'film'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
parser.add_argument('--attack', type=str, default='meta', choices=['pgd', 'meta', 'random', 'nettack', 'dice'])
parser.add_argument('--dropout', type=float, default=0.6, help='Dr0opout rate.')
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--combine', type=str, default='mul', help='{add, mul}}')
parser.add_argument('--hid', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.') # 0.005
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.') # 300
parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameters.')
parser.add_argument('--feature_flip', type=int, default=0, help='Random flip features.')
args = parser.parse_args()
result_list = []
OUT_PATH = "../results/"

def train(net, optimizer, data):
    net.train()
    optimizer.zero_grad()
    output, output_list = net(data.x, data)
    loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss = loss_train
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss, acc

def val(net, data):
    net.eval()
    output, output_list = net(data.x, data)
    loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val

def test(net, data):
    net.eval()
    output, output_list = net(data.x, data)
    loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])

    # Calculate AUC
    auc_test = compute_auc(output[data.test_mask], data.y[data.test_mask])

    # Calculate F1-score
    f1_test = compute_f1(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test, auc_test, f1_test

acc_all = []
auc_all = []
f1_all = []
for i in range(0, 1):

    args.cuda = torch.cuda.is_available()
    print('cuda:%s' % args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make sure you use the same data splits as you generated attacks
    np.random.seed(args.seed)
    if args.ptb_rate == 0:
        args.attack = "no"
    # load original dataset (to get clean features and labels)
    if args.dataset in ['citeseer', 'cora', 'citeseer', 'polblogs', 'cora_ml']:
        data = Dataset(root='../data', name=args.dataset, setting='nettack')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif args.dataset in ['reddit']:
        adj = sp.load_npz('../new_dataset/reddit_graph.npz').tocsr()
        reddit_data = np.load(os.path.join('../new_dataset/reddit_data.npz'))
        features = reddit_data["feature"]
        features = sp.csr_matrix(features)
        labels = reddit_data["label"]
        # tarin/val/test indices
        node_types = reddit_data["node_types"]
        train_mask = (node_types == 1)
        val_mask = (node_types == 2)
        test_mask = (node_types == 3)
        idx_train = np.where(train_mask)[0]
        idx_val = np.where(val_mask)[0]
        idx_test = np.where(test_mask)[0]
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_new_data(args.dataset)
        features = sp.csr_matrix(features)
    # if args.dataset == 'pubmed':
    #     idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
    #                                                       val_size=0.1, test_size=0.8,
    #                                                       stratify=encode_onehot(labels),
    #                                                       seed=15)
    """
           Load the data after the attack.
           """
    if args.attack == 'no':
        perturbed_adj = adj
        # nettack
        # json_file = osp.join('../nettack_modified_graph_data/{}_nettacked_nodes.json'.format(args.dataset))
        # with open(json_file, 'r') as f:
        #     idx = json.loads(f.read())
        # # a = idx["idx_train"]
        # b = idx["attacked_test_nodes"]
        # idx_test = np.array(b)
    else:
        if args.dataset in ['citeseer', 'cora' 'polblogs', 'cora_ml']:
            if args.attack == 'meta':
                if args.ptb_rate == 0.1 or args.ptb_rate == 0.2:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.1f.npz" % (args.dataset, args.ptb_rate)
                else:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.2f.npz" % (args.dataset, args.ptb_rate)
            if args.attack == 'nettack':
                perturbed_data_file = "../nettack_modified_graph_data/%s_nettack_adj_%.1f.npz" % (
                    args.dataset, args.ptb_rate)
                json_file = osp.join('../nettack_modified_graph_data/{}_nettacked_nodes.json'.format(args.dataset))
                with open(json_file, 'r') as f:
                    idx = json.loads(f.read())
                # a = idx["idx_train"]
                b = idx["attacked_test_nodes"]
                idx_test = np.array(b)

            if args.attack == 'random':
                perturbed_data_file = "../OtherAttack/random_modified_graph_data/%s_random_adj_%.0f.npz" % (
                    args.dataset, args.ptb_rate)
            if args.attack == 'pgd':
                perturbed_data_file = "../OtherAttack/pgd_modified_graph_data/%s_pgd_adj_%.0f.npz" % (
                    args.dataset, args.ptb_rate)
            if args.attack == 'dice':
                perturbed_data_file = "../OtherAttack/dice_modified_graph_data/%s_dice_adj_%.0f.npz" % (
                    args.dataset, args.ptb_rate)

            print("perturbed data file is:", perturbed_data_file)
            perturbed_adj = sp.load_npz(perturbed_data_file)
        else:
            if args.attack == 'meta':
                if args.ptb_rate == 0.1 or args.ptb_rate == 0.2:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.1f.npy" % (args.dataset, args.ptb_rate)
                else:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.2f.npy" % (args.dataset, args.ptb_rate)
            if args.attack == 'nettack':
                perturbed_data_file = "../nettack_modified_graph_data/%s_nettack_adj_%.1f.npz" % (
                    args.dataset, args.ptb_rate)

            print("perturbed data file is:", perturbed_data_file)
            perturbed_adj = np.load(perturbed_data_file)
            perturbed_adj = sp.csr_matrix(perturbed_adj)

    nclass = max(labels) + 1
    file_name = "citeseer/edge_diff_matrix_0.05.npy"
    edge_diff_matrix = np.load(file_name)
    file_name = "citeseer/edge_diff_matrix_high_0.05.npy"
    edge_diff_matrix_high = np.load(file_name)
    features = feature_flip(args.dataset, args.feature_flip, features, idx_train)
    data = process(perturbed_adj, features, labels, idx_train, idx_val, idx_test, args.alpha,edge_diff_matrix, edge_diff_matrix_high)

    net = DeepGCN(features.shape[1], args.hid, nclass, edge_diff_matrix, edge_diff_matrix_high,
                                         dropout=args.dropout,
                                         combine=args.combine,
                                         nlayer=args.nlayer)
    net = net.to(device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
    # train
    best_acc = 0
    patient = 0
    best_loss = 1e10
    import time
    s = time.time()
    for epoch in range(args.epochs):

        train_loss, train_acc = train(net, optimizer, data)
        val_loss, val_acc = val(net, data)
        test_loss, test_acc, auc_test, f1_test = test(net, data)

        print('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f | test acc %.3f.' %
                      (epoch, train_loss, train_acc, val_loss, val_acc, test_acc))
        # save model
        if best_acc <= val_acc:
            best_acc = val_acc
            patient = 0
            torch.save(net.state_dict(), OUT_PATH + 'checkpoint-best-acc' + str(args.nlayer) + str(args.dataset) + '.pkl')
        # if best_loss > val_loss:
        #     best_loss = val_loss
        #     torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-acc'+str(args.nlayer) + str(args.dataset) + '.pkl')
        else:
            patient = patient + 1

        if patient >= 20:
            # # torch.save(net.state_dict(),
            #            OUT_PATH + 'checkpoint-best-acc' + str(args.nlayer) + str(args.dataset) + '.pkl')
            print("======early stop=======")
            break
    e = time.time()
    print("avr epoch time:", (e-s)/args.epochs)
    # pick up the best model based on val_acc, then do test
    net.load_state_dict(torch.load(OUT_PATH + 'checkpoint-best-acc' + str(args.nlayer) + str(args.dataset) + '.pkl'))

    val_loss, val_acc = val(net, data)
    test_loss, test_acc, test_auc, test_f1 = test(net, data)

    print("-" * 50)
    print("Vali set results: loss %.3f, acc %.3f." % (val_loss, val_acc))
    print("Test set results: loss %.3f, acc %.3f." % (test_loss, test_acc))
    print("=" * 50)
    acc_all.append(test_acc.cpu().numpy() * 100)
    auc_all.append(test_auc * 100)
    f1_all.append(test_f1 * 100)
print('avr:', np.average(acc_all))
print('std:', np.std(acc_all))
print('var:', np.var(acc_all))

print('auc:', np.average(auc_all))
print('f1:', np.average(f1_all))



