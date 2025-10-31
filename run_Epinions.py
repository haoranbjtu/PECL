import time
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pickle
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.tools import index_generator, parse_minibatch_LastFM
from model import RWGNN_lp
from utils.pytorchtools import EarlyStopping
from utils.data import load_Ciao_data
from utils.tools import index_generator, parse_minibatch_Ciao_homo, RMSELoss
from model.RWGNN_lp import RWGNN_lp

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = '%m/%d/%Y %H:%M:%S %p'
logging.basicConfig(filename='epinion_train.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
# Params
EPS = 1e-10

# num_user = 22166
# num_item = 296277
num_user = 7411
num_item = 8728
# num_user = 2674  # 冷启动数据集
# num_item = 7924
num_ntype = 2
dropout_rate = 0
lr = 0.001
weight_decay = 0.001
expected_metapaths = [
    [(0, 1, 0), (0, 1, 0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0)],  # 0.33 0.27 0.40
    [(1, 0, 1), (1, 0, 1, 0, 1), (1, 1, 0, 1), (1, 1), (1, 0, 1, 1)]  # 0.66  0.26  0.07
]
etypes_lists = [[[0, 1], [0, 1, 0, 1], [0, None, 1], [0, None, None, 1]],  # item 顺序是根据关系多少确定  tools.py 212
                [[1, 0], [1, 0, 1, 0], [None, 1, 0], [None], [1, 0, None]]]  # user  None 不会旋转 刚好是我想要的 社交关系旋转无意义
metapath_list = [
    [['rated-by', 'rated'],  # 0.19
     ['rated-by', 'rated', 'rated-by', 'rated'],  # 0.15
     ['rated-by', 'trust', 'rated']],  # 0.32
     # ['rated-by', 'trust', 'trust', 'rated']],  # 0.33
    [['rated', 'rated-by'],  # 0.50
     ['rated', 'rated-by', 'rated', 'rated-by'],  # 0.3373
     ['trust', 'rated', 'rated-by'],  # 0.07
     ['trust']]]
     # ['rated', 'rated-by', 'trust']],  # 0.07


pathlen_lists = [[1, 2, 3], [1, 2, 3]]


# 取正常完整数据集
outputdir = './data/preprocessed/epinions_dgl_all'
with open(outputdir + '/homo_small_dataset.pkl', 'rb') as f:
    train_df = pickle.load(f)
    test_df = pickle.load(f)
    val_df = pickle.load(f)
    trust_df = pickle.load(f)
    train_g = pickle.load(f)
    test_g = pickle.load(f)
    val_g = pickle.load(f)


def run_model_Ciao(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):

    # type_mask, train_val_test_ratings = load_Ciao_data(prefix='data/preprocessed/Epinions_processed')
    dim = num_item + num_user
    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_item:] = 1
    # 自己做图采样 取代上面这句
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:  # all id vector 全1
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))  # 独热
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    train_ratings = train_df[
        ['iid', 'uid', 'rating']].to_numpy()  # 数据第0个是user 但我的定义里0代表item 做一下替换
    val_ratings = val_df[['iid', 'uid', 'rating']].to_numpy()
    test_ratings = test_df[['iid', 'uid', 'rating']].to_numpy()

    test_loss_list = []
    test_mae_list = []
    for _ in range(repeat):
        net = RWGNN_lp(
            [3, 3], 5, train_g.to(device), pathlen_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type,
            dropout_rate)  # 边的类型两种 10 01  # [3, 4], 2,
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_ratings))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_ratings), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):  # 一批次一批次的id输入模型计算 有分batch
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                rating_label = torch.tensor(train_ratings[train_pos_idx_batch][:, 2].tolist()).to(
                    device)  # 是batch的评分  label的评分 与旋转路径相关

                users_batch = torch.tensor(train_ratings[train_pos_idx_batch][:, 1].tolist()).to(
                    device)
                items_batch = torch.tensor(train_ratings[train_pos_idx_batch][:, 0].tolist()).to(
                    device)
                train_pos_g_lists, train_pos_indices_lists, train_rating_indices_lists, train_pos_idx_batch_mapped_lists = \
                    parse_minibatch_Ciao_homo(train_g, pathlen_lists, train_pos_idx_batch, train_ratings,
                                         args.num_paths_per_node, device, offset=num_item, restart_prob=args.restart_prob)

                t1 = time.time()
                dur1.append(t1 - t0)
                [pos_embedding_user, pos_embedding_artist], _, rating_predictions, _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_rating_indices_lists,
                     train_pos_idx_batch_mapped_lists, (items_batch, users_batch)))  # 对于一个batch的

                # embedding_item = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])  # 因为这次id设置是item user 所以和程序相反
                # embedding_user = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                # rating_predictions = torch.bmm(embedding_item, embedding_user)  # 评分公式1 内积  # rating_label需要整成迭代器
                criterion = RMSELoss()
                rmse, mae = criterion(rating_label, rating_predictions.view(-1))
                train_loss = rmse+mae   # + torch.mean(mae)  # rmse  记得正则化

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | RMSE {:.4f} | MAE {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, torch.sqrt(rmse), torch.mean(mae), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
                    logging.info(
                        'Epoch {:05d} | Iteration {:05d} | RMSE {:.4f} | MAE {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, torch.sqrt(rmse), torch.mean(mae), np.mean(dur1), np.mean(dur2),
                            np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            mae_list = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_idx_batch.sort()
                    # if len(val_ratings[val_idx_batch][:, 3].tolist()) < batch_size:
                    #    continue
                    rating_label = torch.tensor(val_ratings[val_idx_batch][:, 2].tolist()).to(
                        device)  # 是batch的评分  label的评分 与旋转路径相关

                    val_g_lists, val_indices_lists, val_rating_indices_lists, val_idx_batch_mapped_lists = \
                        parse_minibatch_Ciao_homo(val_g, pathlen_lists, val_idx_batch, val_ratings,
                                             args.num_paths_per_node, device, offset=num_item, restart_prob=args.restart_prob)

                    users_batch = torch.tensor(val_ratings[val_idx_batch][:, 1].tolist()).to(
                        device)
                    items_batch = torch.tensor(val_ratings[val_idx_batch][:, 0].tolist()).to(
                        device)
                    [pos_embedding_user, pos_embedding_artist], _, rating_predictions, _ = net(
                        (val_g_lists, features_list, type_mask, val_indices_lists, val_rating_indices_lists,
                         val_idx_batch_mapped_lists, (items_batch, users_batch)))  # 对于一个batch的
                    """
                    embedding_item = pos_embedding_user.view(-1, 1,
                                                             pos_embedding_user.shape[1])  # 因为这次id设置是item user 所以和程序相反
                    embedding_user = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                    rating_predictions = torch.bmm(embedding_item, embedding_user)  # 评分公式1 内积  # rating_label需要整成迭代器
                    """
                    criterion = RMSELoss()
                    rmse, mae = criterion(rating_label, rating_predictions.view(-1))
                    val_loss.append(torch.sqrt(rmse))  # rmse  记得正则化
                    mae_list.append(torch.mean(mae))
                val_loss = torch.mean(torch.tensor(val_loss))
                mae = torch.mean(torch.tensor(mae_list))
                t_end = time.time()
                # print validation info
                print('Epoch {:05d} | Val_Loss {:.4f} | MAE {:.4f} | Time(s) {:.4f}'.format(
                    epoch, torch.mean(val_loss), mae, t_end - t_start))
                logging.info('Epoch {:05d} | Val_Loss {:.4f} | MAE {:.4f} | Time(s) {:.4f}'.format(
                    epoch, torch.mean(val_loss), mae, t_end - t_start))
                # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                logging.info('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_ratings), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        test_loss = []
        mae_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_idx_batch.sort()
                # if len(test_ratings[test_idx_batch][:, 3].tolist()) < batch_size:
                #    continue
                rating_label = torch.tensor(test_ratings[test_idx_batch][:, 2].tolist()).to(
                    device)  # 是batch的评分  label的评分 与旋转路径相关

                test_g_lists, test_indices_lists, test_rating_indices_lists, test_idx_batch_mapped_lists = \
                    parse_minibatch_Ciao_homo(test_g, pathlen_lists, test_idx_batch, test_ratings,
                                         args.num_paths_per_node, device, offset=num_item, restart_prob=args.restart_prob)

                users_batch = torch.tensor(test_ratings[test_idx_batch][:, 1].tolist()).to(
                    device)
                items_batch = torch.tensor(test_ratings[test_idx_batch][:, 0].tolist()).to(
                    device)
                [pos_embedding_user, pos_embedding_artist], _, rating_predictions, _ = net(
                    (test_g_lists, features_list, type_mask, test_indices_lists, test_rating_indices_lists,
                     test_idx_batch_mapped_lists, (items_batch, users_batch)))  # 对于一个batch的
                """
                embedding_item = pos_embedding_user.view(-1, 1,
                                                         pos_embedding_user.shape[1])  # 因为这次id设置是item user 所以和程序相反
                embedding_user = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                rating_predictions = torch.bmm(embedding_item, embedding_user)  # 评分公式1 内积  # rating_label需要整成迭代器
                """
                criterion = RMSELoss()
                rmse, mae = criterion(rating_label, rating_predictions.view(-1))
                test_loss.append(torch.sqrt(rmse))  # rmse  记得正则化
                mae_list.append(torch.mean(mae))
        test_loss = torch.mean(torch.tensor(test_loss))
        mae = torch.mean(torch.tensor(mae_list))
        # auc = roc_auc_score(y_true_test, y_proba_test)
        # ap = average_precision_score(y_true_test, y_proba_test)
        print('Rating Prediction Test')
        print('RMSE = {}, MAE = {}'.format(torch.mean(test_loss), mae))
        logging.info('Rating Prediction Test')
        logging.info('RMSE = {}, MAE = {}'.format(torch.mean(test_loss), mae))
        test_loss_list.append(torch.mean(test_loss))
        test_mae_list.append(mae)
    
    print('----------------------------------------------------------------')
    print('Rating Prediction Tests Summary')
    print('RMSE_mean = {}, RMSE_std = {}'.format(np.mean(test_loss_list), np.std(test_loss_list)))
    print('MAE_mean = {}, MAE_std = {}'.format(np.mean(test_mae_list), np.std(test_mae_list)))
    logging.info('----------------------------------------------------------------')
    logging.info('Rating Prediction Tests Summary')
    logging.info('RMSE_mean = {}, RMSE_std = {}'.format(np.mean(test_loss_list), np.std(test_loss_list)))
    logging.info('MAE_mean = {}, MAE_std = {}'.format(np.mean(test_mae_list), np.std(test_mae_list)))
    # print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


def test(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
         num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    # type_mask, train_val_test_ratings = load_Ciao_data(prefix='data/preprocessed/Epinions_processed')
    dim = num_item + num_user
    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_item:] = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:  # all id vector 全1
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))  # 独热
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    net = RWGNN_lp(
        [4, 4], 5, train_g, pathlen_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type,
        dropout_rate)  # 边的类型两种 10 01  # [3, 4], 2,
    net.to(device)
    test_ratings = test_df[['iid', 'uid', 'rating']].to_numpy()

    test_loss_list = []
    test_mae_list = []
    test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_ratings), shuffle=False)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
    net.eval()
    test_loss = []
    mae_list = []
    with torch.no_grad():
        for iteration in range(test_idx_generator.num_iterations()):
            # forward
            test_idx_batch = test_idx_generator.next()
            test_idx_batch.sort()
            # if len(test_ratings[test_idx_batch][:, 3].tolist()) < batch_size:
            #    continue
            rating_label = torch.tensor(test_ratings[test_idx_batch][:, 2].tolist()).to(
                device)  # 是batch的评分  label的评分 与旋转路径相关

            test_g_lists, test_indices_lists, test_rating_indices_lists, test_idx_batch_mapped_lists = \
                parse_minibatch_Ciao_homo(test_g, pathlen_lists, test_idx_batch, test_ratings,
                                     args.num_paths_per_node, device, offset=num_item, restart_prob=args.restart_prob)
            # 测试集不能用评分
            """
            test_rating_indices_lists = [[
                torch.where(test_rating_indice != 3, torch.tensor(3).to(test_rating_indice.device),
                            test_rating_indice) for test_rating_indice in test_rating_indices] for test_rating_indices
                in test_rating_indices_lists]
            """
            users_batch = torch.tensor(test_ratings[test_idx_batch][:, 1].tolist()).to(
                device)
            items_batch = torch.tensor(test_ratings[test_idx_batch][:, 0].tolist()).to(
                device)
            [pos_embedding_user, pos_embedding_artist], _, rating_predictions, _ = net(
                (test_g_lists, features_list, type_mask, test_indices_lists, test_rating_indices_lists,
                 test_idx_batch_mapped_lists, (items_batch, users_batch)))  # 对于一个batch的
            """
            embedding_item = pos_embedding_user.view(-1, 1,
                                                     pos_embedding_user.shape[1])  # 因为这次id设置是item user 所以和程序相反
            embedding_user = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
            rating_predictions = torch.bmm(embedding_item, embedding_user)  # 评分公式1 内积  # rating_label需要整成迭代器
            """
            criterion = RMSELoss()
            rmse, mae = criterion(rating_label, rating_predictions.view(-1))
            test_loss.append(torch.sqrt(rmse))  # rmse  记得正则化
            mae_list.append(torch.mean(mae))
        test_loss = torch.mean(torch.tensor(test_loss))
        mae = torch.mean(torch.tensor(mae_list))
        # auc = roc_auc_score(y_true_test, y_proba_test)
        # ap = average_precision_score(y_true_test, y_proba_test)
        print('Rating Prediction Test')
        print('RMSE = {}, MAE = {}'.format(torch.mean(test_loss), mae))
        logging.info('Rating Prediction Test')
        logging.info('RMSE = {}, MAE = {}'.format(torch.mean(test_loss), mae))
        test_loss_list.append(torch.mean(test_loss))
        test_mae_list.append(mae)
    print('----------------------------------------------------------------')
    print('Rating Prediction Tests Summary')
    print('RMSE_mean = {}, RMSE_std = {}'.format(np.mean(test_loss_list), np.std(test_loss_list)))
    print('MAE_mean = {}, MAE_std = {}'.format(np.mean(test_mae_list), np.std(test_mae_list)))
    logging.info('----------------------------------------------------------------')
    logging.info('Rating Prediction Tests Summary')
    logging.info('RMSE_mean = {}, RMSE_std = {}'.format(np.mean(test_loss_list), np.std(test_loss_list)))
    logging.info('MAE_mean = {}, MAE_std = {}'.format(np.mean(test_mae_list), np.std(test_mae_list)))
    # print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))
    # print(rating_predictions)
    # print(rating_label)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RWGNN testing for the epinions dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=64, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='lstm', help='Type of the aggregator. Default is RotatE1.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')  # 早停
    ap.add_argument('--batch-size', type=int, default=512, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=20, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='Epinions', help='Postfix for the saved model and result. Default is Ciao.')
    ap.add_argument('--num-paths-per-node', default=10, help='num_paths_per_node. Default is 12.')
    ap.add_argument('--restart-prob', default=0.2, help='num_paths_per_node. Default is 12.')  # 20


    args = ap.parse_args()
    logging.info(
        f"feats_type={args.feats_type}, hidden_dim={args.hidden_dim}, num_heads={args.num_heads}, attn_vec_dim={args.attn_vec_dim}, rnn_type={args.rnn_type}, epoch={args.epoch}, \
                             batch_size={args.batch_size}, samples={args.samples}, save_postfix={args.save_postfix}, dropout_rate={dropout_rate}, lr={lr}, weight_decay={weight_decay}")
    run_model_Ciao(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)  # run_model_Ciao

