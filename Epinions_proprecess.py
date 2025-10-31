import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import pickle
import networkx as nx
import utils.preprocess
from sklearn.model_selection import train_test_split


from utils.data import load_glove_vectors
from utils.data import creat_rating_dict
# 按照MAGNN的处理方式建图和生成基于元路径的随机游走路径 先跑通，后续可调整，部分用dgl实现

save_prefix = 'data/preprocessed/Epinions_processed/'
num_ntypes = 2

ratings_df = pd.read_csv('data/raw data/epinions/ratings.txt', sep=' ', header=None, names=['user', 'item', 'rating'], keep_default_na=False, encoding='utf-8')
trust_df = pd.read_csv('RW_GNN_CF/data/raw data/epinions/trust.txt', sep=' ', header=None, names=['u1', 'u2'], keep_default_na=False, encoding='utf-8')

glove_dim = 50
# glove_vectors = load_glove_vectors(dim=glove_dim) 加载glove词嵌入

# 划分数据集 随机 或者 按时间 70 10 20
train_df = ratings_df.sample(frac=0.7, random_state=123)
test_val_df = ratings_df[~ratings_df.index.isin(train_df.index)]
test_df = test_val_df.sample(frac=2/3, random_state=123)
val_df = test_val_df[~test_val_df.index.isin(test_df.index)]


# build the adjacency matrix for the graph consisting of users and items
# 0 for items, 1 for users
num_item = max(ratings_df['item'])
num_user = max(ratings_df['user'])
dim = num_item + num_user
type_mask = np.zeros((dim), dtype=int)
type_mask[max(ratings_df['item']):] = 1

item_id_mapping = {i+1: i for i in range(num_item)}
user_id_mapping = {row['item']: i + num_item for i, row in ratings_df.iterrows()}
print(num_item)

np.save(save_prefix + 'node_types.npy', type_mask)


# output samples for training, validation and testing

# 评分预测任务不需要负采样
train_df['user'] = train_df['user'] - 1
train_df['item'] = train_df['item'] - 1
val_df['user'] = val_df['user'] - 1
val_df['item'] = val_df['item'] - 1
test_df['user'] = test_df['user'] - 1
test_df['item'] = test_df['item'] - 1


np.savez(save_prefix + 'train_val_test_ratings.npz',
         train_ratings=train_df,
         val_ratings=val_df,
         test_ratings=test_df)


