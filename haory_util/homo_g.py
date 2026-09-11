import numpy as np
import pandas as pd
import torch
import dgl
import pickle
import os
script_path = os.path.dirname(os.path.abspath(__file__))
inputdir = script_path + '/../data/raw data/ml-1m'
outputdir = script_path + '/../data/preprocessed/ml_dgl_all'
# outputdir = '../data/preprocessed/epinions_dgl_all'
# inputdir = '../data/raw data/epinions'


def build_train_graph(g, train_indices):
    train_g = dgl.edge_subgraph(g,
                                train_indices, relabel_nodes=False)

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g


rate_f = pd.read_csv(inputdir + '/ratings.dat', sep='::',
                     names=['uid', 'iid', 'rating', 'date'])

trust_f = pd.read_csv(inputdir + '/trust.txt', sep=' ', names=['user1', 'user2']) - 1
"""
rate_f = pd.read_csv(inputdir + '/ratings.txt', sep=',',
                     names=['uid', 'iid', 'rating'])
trust_f = pd.read_csv(inputdir + '/trust.txt', sep=' ', names=['user1', 'user2']) - 1
"""
# rate_f = pd.read_csv(outputdir + '/cold_rating.csv', sep=' ', names=['uid', 'iid', 'rating', '_1', '_2', '_3', '_4', '_5'])  # 冷启动
# trust_f = pd.read_csv(outputdir + '/cold_trust.csv', sep=' ', names=['user1', 'user2']) - 1

print(rate_f)
num_user = max(rate_f['uid'])
num_item = max(rate_f['iid'])
print(num_user)
print(num_item)
rate_f['uid'] += num_item - 1
rate_f['iid'] -= 1
pos_df = rate_f.sort_values(['date'])  # 按时间划分训练集
from sklearn.utils import shuffle
# pos_df = shuffle(rate_f)

train_df = pos_df[:int(0.8 * len(pos_df))]
val_df = pos_df[int(0.8 * len(pos_df)): int(0.9 * len(pos_df))]
# val_df = pos_df[int(0.8 * len(pos_df)):]

test_df = pos_df[int(0.9 * len(pos_df)):len(pos_df)]
pos_df['train_mask'] = np.ones((len(pos_df),), dtype=bool)
pos_df['test_mask'] = np.ones((len(pos_df),), dtype=bool)
pos_df['val_mask'] = np.ones((len(pos_df),), dtype=bool)
pos_df['train_mask'][:int(0.8 * len(pos_df))] = True
pos_df['train_mask'][int(0.8 * len(pos_df)):] = False
pos_df['val_mask'][int(0.8 * len(pos_df)):] = True
pos_df['val_mask'][:int(0.8 * len(pos_df)):int(0.9 * len(pos_df))] = False
pos_df['val_mask'][int(0.9 * len(pos_df)):] = False
pos_df['test_mask'][:int(0.9 * len(pos_df))] = False
pos_df['test_mask'][int(0.9 * len(pos_df)):len(pos_df)] = True
train_indices = pos_df['train_mask'].to_numpy().nonzero()[0]
val_indices = pos_df['val_mask'].to_numpy().nonzero()[0]
test_indices = pos_df['test_mask'].to_numpy().nonzero()[0]

pos_u_l = pos_df['uid'].tolist()
pos_i_l = pos_df['iid'].tolist()
trust_l1 = trust_f['user1'].tolist()
trust_l2 = trust_f['user2'].tolist()
trust_f['rating'] = 0
src = torch.tensor(pos_u_l + pos_i_l + trust_l1 + trust_l2)
dst = torch.tensor(pos_i_l + pos_u_l + trust_l2 + trust_l1)
# src = torch.tensor(pos_u_l + pos_i_l)
# dst = torch.tensor(pos_i_l + pos_u_l)
g = dgl.graph((src, dst))
g.edata['rating'] = torch.tensor(pos_df['rating'].tolist() + pos_df['rating'].tolist() + trust_f['rating'].tolist() + trust_f['rating'].tolist(), dtype=torch.long)  # trust的评分是0
# g.edata['time'] = torch.tensor(pos_df['date'].tolist() + pos_df['date'].tolist() + trust_f['rating'].tolist() + trust_f['rating'].tolist(), dtype=torch.long)  # trust的评分是0
g.edata['time'] = torch.tensor(torch.zeros_like(g.edata['rating']), dtype=torch.long)  # Epinions没有时间戳
# g.edata['rating'] = torch.tensor(pos_df['rating'].tolist() + pos_df['rating'].tolist(), dtype=torch.long)  # trust的评分是0
# g.edata['time'] = torch.tensor(pos_df['date'].tolist() + pos_df['date'].tolist(), dtype=torch.long)  # trust的评分是0
num_rating = len(pos_df)
num_trust = len(trust_f)
train_indices_g = torch.tensor(train_indices.tolist() + (train_indices+num_rating).tolist() + (trust_f.index + num_rating*2).tolist() + (trust_f.index + num_rating*2 + num_trust).tolist())
val_indices_g = torch.tensor(val_indices.tolist() + (val_indices+num_rating).tolist() + (trust_f.index + num_rating*2).tolist() + (trust_f.index + num_rating*2 + num_trust).tolist())
test_indices_g = torch.tensor(test_indices.tolist() + (test_indices+num_rating).tolist() + (trust_f.index + num_rating*2).tolist() + (trust_f.index + num_rating*2 + num_trust).tolist())
# train_indices_g = torch.tensor(train_indices.tolist() + (train_indices+num_rating).tolist())
# val_indices_g = torch.tensor(val_indices.tolist() + (val_indices+num_rating).tolist())
# test_indices_g = torch.tensor(test_indices.tolist() + (test_indices+num_rating).tolist())
train_g = build_train_graph(g, train_indices_g)
val_g = build_train_graph(g, val_indices_g)
test_g = build_train_graph(g, test_indices_g)


# Allow this script to run independently on a fresh checkout.
os.makedirs(outputdir, exist_ok=True)
with open(outputdir + '/homo_dataset.pkl', 'wb') as f:
    pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(trust_f, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_g, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_g, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_g, f, pickle.HIGHEST_PROTOCOL)

"""
with open(outputdir + '/homo_cold_dataset.pkl', 'wb') as f:
    pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(trust_f, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_g, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_g, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(val_g, f, pickle.HIGHEST_PROTOCOL)
"""
