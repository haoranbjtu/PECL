import random
import pickle
import scipy.io as scio
import scipy.sparse as ssp
import numpy as np
import pandas as pd

import torch
import dgl

from builder import PandasGraphBuilder

inputdir = '../data/raw data/ciao'
outputdir = '../data/preprocessed/ciao_dgl_all'

rate_f = np.loadtxt(inputdir + '/rating_with_timestamp.txt', dtype=np.int32, delimiter=' ')
# names=['user', 'item', 'genreID', 'rating', 'helpfulness', 'date'],
rate_line_id = 3
no_helpfulness = False
trust_f = np.loadtxt(inputdir + '/trust.txt', delimiter=' ')
rate_list = []
trust_list = []

user_count = 0
item_count = 0

for s in rate_f:  # 所有数据
    uid = s[0]
    iid = s[1]
    label = s[rate_line_id]
    if uid > user_count:
        user_count = uid
    if iid > item_count:
        item_count = iid
    if not no_helpfulness:
        helpfulness = s[2]
        rate_list.append([uid, iid, label, helpfulness])
    else:
        rate_list.append([uid, iid, label])

pos_list = []  # 只有三元组


for i in range(len(rate_list)):
    pos_list.append((rate_list[i][1], rate_list[i][0], rate_list[i][2], rate_list[i][3]))
random.shuffle(pos_list)
pos_df = pd.DataFrame(pos_list, columns=['iid', 'uid', 'label', 'helpfulness'])
pos_df['uid'] -= 1  # id从0开始
pos_df['iid'] -= 1



neg_candidates = []
counter = 0
for i in range(num_user):
    for j in range(num_artist):
        if counter < len(user_artist):
            if i == user_artist[counter, 0] and j == user_artist[counter, 1]:
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)

idx = np.random.choice(len(neg_candidates), len(val_idx) + len(test_idx), replace=False)
val_neg_candidates = neg_candidates[sorted(idx[:len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx):])]

train_user_artist = user_artist[train_idx]
train_neg_candidates = []
counter = 0
for i in range(num_user):
    for j in range(num_artist):
        if counter < len(train_user_artist):
            if i == train_user_artist[counter, 0] and j == train_user_artist[counter, 1]:
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

np.savez(save_prefix + 'train_val_test_neg.npz',
         train_neg_user_artist=train_neg_candidates,
         val_neg_user_artist=val_neg_candidates,
         test_neg_user_artist=test_neg_candidates)
