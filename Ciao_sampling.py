import time
import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
import dgl
import argparse
import pandas as pd

import torch.nn.functional as F
# from sklearn.metrics import roc_auc_score, average_precision_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_Ciao_data
from utils.tools import index_generator, parse_minibatch_Ciao
from model.RWGNN_lp import RWGNN_lp

EPS = 1e-10


def init_weight(userNum, itemNum, hide_dim):
    initializer = nn.init.xavier_uniform_
    embedding_dict = nn.ParameterDict({
        'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
        'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
    })
    return embedding_dict


# 取正常完整数据集
outputdir = './data/preprocessed/ciao_dgl_all'
with open(outputdir + '/dataset.pkl', 'rb') as f:
    train_df = pickle.load(f)
    test_df = pickle.load(f)
    val_df = pickle.load(f)
    trust_df = pickle.load(f)
    train_g = pickle.load(f)
    test_g = pickle.load(f)
    val_g = pickle.load(f)
    mean_rating = pickle.load(f)

# Params
num_ntype = 2
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_lists = [[[0, 1], [0, 1, 0, 1], [0, None, 1]],  # item 顺序是根据关系多少确定  tools.py 212
                [[1, 0], [1, 0, 1, 0], [1, 0, None, 1, 0], [None]]]  # user  None 不会旋转 刚好是我想要的 社交关系旋转无意义
# 原版用None，即表示不参与旋转 edge编号：0: 0, 1; 1: 1, 0; 2: 1, 1
use_masks = [[False, False, False],
             [False, False, False, False]]  #
no_masks = [[False] * 3, [False] * 4]
num_user = 2378
num_item = 16861
expected_metapaths = [
    [(0, 1, 0), (0, 1, 0, 1, 0), (0, 1, 1, 0)],
    [(1, 0, 1), (1, 0, 1, 0, 1), (1, 0, 1, 1, 0, 1), (1, 1)]
]
metapath_list = [
    [['rated-by', 'rated'],
     ['rated-by', 'rated', 'rated-by', 'rated'],
     ['rated-by', 'trust', 'rated']],
    [['rated', 'rated-by'],
     ['rated', 'rated-by', 'rated', 'rated-by'],
     ['rated', 'rated-by', 'trust', 'rated', 'rated-by'],
     ['trust']],
]
# 图中的信任关系是单向

metapath_dict = {
    ('rated', 'rated-by'): (1, 0, 1),
    ('rated', 'rated-by', 'rated', 'rated-by'): (1, 0, 1, 0, 1),
    ('trust'): (1, 1),
    ('rated', 'rated-by', 'trust', 'rated', 'rated-by'): (1, 0, 1, 1, 0, 1),
    ('rated-by', 'rated'): (0, 1, 0),
    ('rated-by', 'rated', 'rated-by', 'rated'): (0, 1, 0, 1, 0),
    ('rated-by', 'trust', 'rated'): (0, 1, 1, 0)
}

adjlists_ua, edge_metapath_indices_list_ua, _, type_mask, train_val_test_ratings = load_Ciao_data()  # 3分钟  再加rating内存就会爆
# print(rating_metapath_indices_list)


# 自己做图采样 取代上面这句
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
features_list = []
in_dims = []
feats_type = 0

if feats_type == 0:  # 独热
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


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        dis = (yhat - y + EPS)
        mae = torch.sum(torch.abs(dis), dim=0) / (y.shape[0])
        return torch.sqrt(torch.sum(torch.pow(dis, 2), dim=0) / (y.shape[0])), mae


train_ratings = train_val_test_ratings['train_ratings']
val_ratings = train_val_test_ratings['val_ratings']
test_ratings = train_val_test_ratings['test_ratings']
train_ratings = pd.DataFrame(train_ratings)
train_ratings.columns = ['user', 'item', 'genreID', 'rating', 'helpfulness', 'date']
train_ratings = train_ratings[
    ['item', 'user', 'genreID', 'rating', 'helpfulness', 'date']].to_numpy()  # 数据第0个是user 但我的定义里0代表item 做一下替换
val_ratings = pd.DataFrame(val_ratings)
val_ratings.columns = ['user', 'item', 'genreID', 'rating', 'helpfulness', 'date']
val_ratings = val_ratings[['item', 'user', 'genreID', 'rating', 'helpfulness', 'date']].to_numpy()
test_ratings = pd.DataFrame(test_ratings)
test_ratings.columns = ['user', 'item', 'genreID', 'rating', 'helpfulness', 'date']
test_ratings = test_ratings[['item', 'user', 'genreID', 'rating', 'helpfulness', 'date']].to_numpy()

hidden_dim = 64
num_heads = 8
attn_vec_dim = 128
rnn_type = 'RotatE1'
batch_size = 8  # 太小太慢
patience = 5  # 早停  能够容忍多少个epoch内都没有improvement
save_postfix = 'Ciao'
# neighbor_samples = 10
num_paths_per_node = 12

net = RWGNN_lp(
    [3, 4], 5, train_g, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type,
    dropout_rate)  # 边的类型两种 10 01  # [3, 4], 2,
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

# training loop
net.train()
early_stopping = EarlyStopping(patience=patience, verbose=True,
                               save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
dur1 = []
dur2 = []
dur3 = []
train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_ratings))
val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_ratings), shuffle=False)
t_start = time.time()
# training
net.train()


for iteration in range(train_pos_idx_generator.num_iterations()):  # 一批次一批次的id输入模型计算 有分batch
    # forward
    t0 = time.time()

    train_pos_idx_batch = train_pos_idx_generator.next()
    train_pos_idx_batch.sort()
    # print(train_pos_idx_batch)  # 八条记录  把评分取出来

    train_pos_user_item_batch = train_ratings[train_pos_idx_batch].tolist()
    # print(train_ratings[train_pos_idx_batch].shape)
    # print(train_ratings[train_pos_idx_batch][:, 3].tolist())
    rating_label = torch.tensor(train_ratings[train_pos_idx_batch][:, 3].tolist()).to(device)  # 是batch的评分  旋转使用的评分 与旋转路径相关

    # train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists, train_rating_indices_lists = parse_minibatch_LastFM(
    #     adjlists_ua, edge_metapath_indices_list_ua, train_pos_user_item_batch, device,
    #   neighbor_samples, use_masks, num_item)  # 采样
    # 生成计算图(同构图)，路径，映射

    train_pos_g_lists, train_pos_indices_lists, train_rating_indices_lists, train_pos_idx_batch_mapped_lists = \
        parse_minibatch_Ciao(train_g, batch_size, metapath_list, train_pos_idx_batch, train_ratings, num_paths_per_node, device, offset=num_item)

    t1 = time.time()
    dur1.append(t1 - t0)
    [pos_embedding_user, pos_embedding_artist], _ = net(
        (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_rating_indices_lists, train_pos_idx_batch_mapped_lists))  # 对于一个batch的

    embedding_item = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])  # 因为这次id设置是item user 所以和程序相反
    embedding_user = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)

    # print(embedding_item.shape)  # torch.Size([8, 1, 64])
    # print(embedding_user.shape)  # torch.Size([8, 64, 1])

    rating_predictions = torch.bmm(embedding_item, embedding_user)  # 评分公式1 内积  # rating_label需要整成迭代器
    criterion = RMSELoss()
    # print(rating_predictions.view(-1))
    rmse, mae = criterion(rating_label, rating_predictions)
    # print(rmse)  # 维度问题
    train_loss = torch.mean(torch.pow(rmse, 2))  # rmse  记得正则化
    # pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist) 定义评分预测公式
    # train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
    # train_loss = rmse
    # ground truth需要取边
    # 建全图，用api采样路径，全图取边 和 emb，计算用采样生成的计算图g
    # 已经跑通 可训练  但用dgl的采样更快 虽然是黑盒 但方便取边的参数
    # 或者不用dgl的元路径采样api 但要成功自己把评分信息加进去
    t2 = time.time()
    dur2.append(t2 - t1)

    # autograd
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    t3 = time.time()
    dur3.append(t3 - t2)

    # print training info
    epoch = 1
    if iteration % 100 == 0:
        print(
            'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
