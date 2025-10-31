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

save_prefix = 'data/preprocessed/Ciao_processed/'
num_ntypes = 2

item_type_df = pd.read_csv('data/raw data/ciao/items.txt', sep=',', header=None, names=['item', 'type'], keep_default_na=False, encoding='utf-8')
ratings_df = pd.read_csv('data/raw data/ciao/rating_with_timestamp.txt', sep=' ', header=None, names=['user', 'item', 'genreID', 'rating', 'helpfulness', 'date'], keep_default_na=False, encoding='utf-8')
trust_df = pd.read_csv('data/raw data/ciao/trust.txt', sep=' ', header=None, names=['u1', 'u2'], keep_default_na=False, encoding='utf-8')

glove_dim = 50
# glove_vectors = load_glove_vectors(dim=glove_dim) 加载glove词嵌入

# 划分数据集 随机 或者 按时间 70 10 20
train_df = ratings_df.sample(frac=0.7, random_state=123)
test_val_df = ratings_df[~ratings_df.index.isin(train_df.index)]
test_df = test_val_df.sample(frac=2/3, random_state=123)
val_df = test_val_df[~test_val_df.index.isin(test_df.index)]


# build the adjacency matrix for the graph consisting of users and items
# 0 for items, 1 for users
num_item = max(item_type_df['item'])
num_user = max(ratings_df['user'])
dim = num_item + num_user
type_mask = np.zeros((dim), dtype=int)
type_mask[max(item_type_df['item']):] = 1

item_id_mapping = {i+1: i for i in range(num_item)}
user_id_mapping = {row['item']: i + num_item for i, row in item_type_df.iterrows()}
print(num_item)

"""
adjM = np.zeros((dim, dim), dtype=int)
for _, row in ratings_df.iterrows():
    idx1 = item_id_mapping[row['item']]
    idx2 = user_id_mapping[row['user']]
    adjM[idx1, idx2] = 1  # 双向图
    adjM[idx2, idx1] = 1
for _, row in trust_df.iterrows():
    idx1 = user_id_mapping[row['u1']]
    idx2 = user_id_mapping[row['u2']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

# 获取邻接表
user_item_list = {i: adjM[i + num_item, :num_item].nonzero()[0] for i in range(num_user)}
item_user_list = {i: adjM[i, num_item:].nonzero()[0] for i in range(num_item)}
user_user_list = {i: adjM[i + num_item, num_item:].nonzero()[0] for i in range(num_user)}

# 元路径随机采样
# 0-1-0 item-user-item
i_u_i = []
for u, i_list in user_item_list.items():
    i_u_i.extend([(i1, u, i2) for i1 in i_list for i2 in i_list])
i_u_i = np.array(i_u_i)
i_u_i[:, 1] += num_item # user的编号在item后
sorted_index = sorted(list(range(len(i_u_i))), key=lambda i: i_u_i[i, [0, 2, 1]].tolist())
i_u_i = i_u_i[sorted_index]
# print(i_u_i)

# 1-0-1 user-item-user
u_i_u = []
for i, u_list in item_user_list.items():
    u_i_u.extend([(u1, i, u2) for u1 in u_list for u2 in u_list])
u_i_u = np.array(u_i_u)
u_i_u[:, 0] += num_item
u_i_u[:, 2] += num_item
sorted_index = sorted(list(range(len(u_i_u))), key=lambda i: u_i_u[i, [0, 2, 1]].tolist())
u_i_u = u_i_u[sorted_index]
# print(u_i_u)

# 1-1
u_u = trust_df.to_numpy(dtype=np.int32) - 1  # user原始编码 没有加num_item
sorted_index = sorted(list(range(len(u_u))), key=lambda i: u_u[i].tolist())
u_u = u_u[sorted_index]
# print(u_u)

# 0-1-1-0
i_u_u_i = []
for u1, u2 in u_u:
    i_u_u_i.extend([(i1, u1, u2, i2) for i1 in user_item_list[u1] for i2 in user_item_list[u2]])
i_u_u_i = np.array(i_u_u_i)
i_u_u_i[:, [1, 2]] += num_item
sorted_index = sorted(list(range(len(i_u_u_i))), key=lambda i: i_u_u_i[i, [0, 3, 1, 2]].tolist())
i_u_u_i = i_u_u_i[sorted_index]
# print(i_u_u_i)

u_u += num_item  # 调整后的编号

# 0, 1, 0, 1, 0
i_u_i_u_i = []
for u1, i, u2 in u_i_u:
    if len(user_item_list[u1 - num_item]) == 0 or len(user_item_list[u2 - num_item]) == 0:
        continue
    candidate_u1_list = np.random.choice(len(user_item_list[u1 - num_item]),
                                         int(0.2 * len(user_item_list[u1 - num_item])), replace=False)
    candidate_u1_list = user_item_list[u1 - num_item][candidate_u1_list]
    candidate_u2_list = np.random.choice(len(user_item_list[u2 - num_item]),
                                         int(0.2 * len(user_item_list[u2 - num_item])), replace=False)
    candidate_u2_list = user_item_list[u2 - num_item][candidate_u2_list]
    i_u_i_u_i.extend([(i1, u1, i, u2, i2) for i1 in candidate_u1_list for i2 in candidate_u2_list])
i_u_i_u_i = np.array(i_u_i_u_i)
sorted_index = sorted(list(range(len(i_u_i_u_i))), key=lambda i: i_u_i_u_i[i, [0, 4, 1, 2, 3]].tolist())
i_u_i_u_i = i_u_i_u_i[sorted_index]
print(i_u_i_u_i)

# 1, 0, 1, 0, 1
u_i_u_i_u = []
for i1, u, i2 in i_u_i:
    if len(item_user_list[i1]) == 0 or len(item_user_list[i2]) == 0:
        continue
    candidate_i1_list = np.random.choice(len(item_user_list[i1]),
                                         int(0.2 * len(item_user_list[i1])), replace=False)
    candidate_i1_list = item_user_list[i1][candidate_i1_list]
    candidate_i2_list = np.random.choice(len(item_user_list[i2]),
                                         int(0.2 * len(item_user_list[i2])), replace=False)
    candidate_i2_list = item_user_list[i2][candidate_i2_list]
    u_i_u_i_u.extend([(u1, i1, u, i2, u2) for u1 in candidate_i1_list for u2 in candidate_i2_list])  # u1 u2忘记加num_item
u_i_u_i_u = np.array(u_i_u_i_u)
u_i_u_i_u[:, [0, 4]] += num_item
sorted_index = sorted(list(range(len(u_i_u_i_u))), key=lambda i: u_i_u_i_u[i, [0, 4, 1, 2, 3]].tolist())
u_i_u_i_u = u_i_u_i_u[sorted_index]
print(u_i_u_i_u)


# 1, 0, 1, 1, 0, 1
u_i_u_u_i_u = []  # 计算量太大 文件好几个G 读取太慢
for i1, u1, u2, i2 in i_u_u_i:
    if len(item_user_list[i1]) == 0 or len(item_user_list[i2]) == 0:
        continue
    candidate_i1_list = np.random.choice(len(item_user_list[i1]),
                                         int(0.2 * len(item_user_list[i1])), replace=False)  # 取20%的邻居  否则太多
    candidate_i1_list = item_user_list[i1][candidate_i1_list]
    candidate_i2_list = np.random.choice(len(item_user_list[i2]),
                                         int(0.2 * len(item_user_list[i2])), replace=False)
    candidate_i2_list = item_user_list[i2][candidate_i2_list]
    u_i_u_u_i_u.extend([(u3, i1, u1, u2, i2, u4) for u3 in candidate_i1_list for u4 in candidate_i2_list])  # u3 u4 + nums_item
u_i_u_u_i_u = np.array(u_i_u_u_i_u)
u_i_u_u_i_u[:, [0, 5]] += num_item
sorted_index = sorted(list(range(len(u_i_u_u_i_u))), key=lambda i: u_i_u_u_i_u[i, [0, 4, 1, 2, 3]].tolist())
u_i_u_u_i_u = u_i_u_u_i_u[sorted_index]
print(u_i_u_u_i_u)


expected_metapaths = [
    [(0, 1, 0), (0, 1, 1, 0), (0, 1, 0, 1, 0)],
    [(1, 0, 1), (1, 1), (1, 0, 1, 0, 1), (1, 0, 1, 1, 0, 1)]
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {(0, 1, 0): i_u_i,
                            (0, 1, 1, 0): i_u_u_i,
                            (0, 1, 0, 1, 0): i_u_i_u_i,
                            (1, 0, 1): u_i_u,
                            (1, 1): u_u,
                            (1, 0, 1, 0, 1): u_i_u_i_u,
                            # (1, 0, 1, 1, 0, 1): u_i_u_u_i_u
                            }

# write all things
target_idx_lists = [np.arange(num_item), np.arange(num_user)]  #
offset_list = [0, num_item]  # item id 的offset为0  user的为num_item
for i, metapaths in enumerate(expected_metapaths):  # [(I-U-I), ()]
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:  # 路径起点检验在这类点id范围内
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]  # target节点所有可能的路径
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        # np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]  # 路径终点  再还原id
                neighbors = list(map(str, neighbors))  # 路径中的邻居(即 路径终点 )
                if len(neighbors) > 0:  # 如果有邻居
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')  # 以字符串形式写
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right

scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
np.save(save_prefix + 'node_types.npy', type_mask)

"""
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

train_raing_dict = creat_rating_dict(ratings_df, item_id_mapping, user_id_mapping)  # 提前生成评分dict，采样完可从字典取分数
val_raing_dict = creat_rating_dict(val_df, item_id_mapping, user_id_mapping)
test_raing_dict = creat_rating_dict(test_df, item_id_mapping, user_id_mapping)

np.save(save_prefix + 'train_val_test_ratings_dict.npy',
         train_raing_dict)
         # val_raing_dict,
         # test_raing_dict)
