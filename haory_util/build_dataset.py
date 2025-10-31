import random
import pickle
import scipy.io as scio
import scipy.sparse as ssp
import numpy as np
import pandas as pd

import torch
import dgl

from builder import PandasGraphBuilder

random.seed(5)  # epinions 97  ciao 1 0.9134  2  0.9070
"""
为Ciao和Epinions构建异构图 方便取emb 属性 采样
"""


def build_train_graph(g, train_indices, train_indices_t, etype1, etype_rev1, etype2):
    train_g = dgl.edge_subgraph(g,
                                {etype1: train_indices, etype_rev1: train_indices, etype2: train_indices_t},  relabel_nodes=False)

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix


def build_test_matrix(g, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return test_matrix


if __name__ == '__main__':
    # workdir = '../dataset/ciao_2378_16861'  # change to your workdir
    # workdir = '../dataset/epinions_22166_296277'
    # inputdir = '../data/raw data/ciao'
    # outputdir = '../data/preprocessed/ciao_dgl_all'
    outputdir = '../data/preprocessed/epinions_dgl_all'
    inputdir = '../data/raw data/epinions'

    if inputdir == '../dataset/raw data/epinions':
        rate_f = np.loadtxt(inputdir + '/ratings.txt', delimiter=',', dtype=np.int32)
        rate_line_id = 2
        no_helpfulness = True
    else:
        rate_f = np.loadtxt(inputdir + '/rating_with_timestamp.txt', dtype=np.int32, delimiter=' ')
        # names=['user', 'item', 'genreID', 'rating', 'helpfulness', 'date'],
        rate_line_id = 3
        no_helpfulness = False
    trust_f = np.loadtxt(inputdir + '/trust.txt', delimiter=' ')
    # names=['u1', 'u2'],

    # item_type_df = pd.read_csv('data/raw data/ciao/items.txt', sep=',', header=None, names=['item', 'type'], keep_default_na=False, encoding='utf-8')

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

    if no_helpfulness:
        for i in range(len(rate_list)):
            pos_list.append((rate_list[i][0], rate_list[i][1], rate_list[i][2]))
        random.shuffle(pos_list)
        pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label'])
    else:
        for i in range(len(rate_list)):
            pos_list.append((rate_list[i][0], rate_list[i][1], rate_list[i][2], rate_list[i][3]))
        random.shuffle(pos_list)
        pos_df = pd.DataFrame(pos_list, columns=['uid', 'iid', 'label', 'helpfulness'])
    pos_df['uid'] -= 1  # id从0开始
    pos_df['iid'] -= 1
    print(pos_df)

    """
    pos_df['boun3'] = pos_df['label']
    pos_df['boun4'] = pos_df['label']
    pos_df.loc[(pos_df['label'] >= 3), 'boun3'] = 1
    pos_df.loc[(pos_df['label'] < 3), 'boun3'] = 0
    pos_df.loc[(pos_df['label'] >= 4), 'boun4'] = 1
    pos_df.loc[(pos_df['label'] < 4), 'boun4'] = 0
    pos_df.loc[(pos_df['label'] >= 2), 'boun2'] = 1
    pos_df.loc[(pos_df['label'] < 2), 'boun2'] = 0
    pos_df.loc[(pos_df['label'] >= 1), 'boun1'] = 1
    pos_df.loc[(pos_df['label'] < 1), 'boun1'] = 0
    pos_df.loc[(pos_df['label'] >= 5), 'boun5'] = 1
    pos_df.loc[(pos_df['label'] < 5), 'boun5'] = 0
    print(pos_df.mean())
    # print(pos_df)
    """

    train_df = pos_df[:int(0.8 * len(pos_list))]
    val_df = pos_df[int(0.8 * len(pos_df)): int(0.9 * len(pos_df))]
    test_df = pos_df[int(0.9 * len(pos_list)):len(pos_list)]
    # print(pos_list)
    # train_df = train_df.sort_values(axis=0, ascending=True, by='uid')
    pos_df['train_mask'] = np.ones((len(pos_df),), dtype=bool)
    pos_df['test_mask'] = np.ones((len(pos_df),), dtype=bool)
    pos_df['val_mask'] = np.ones((len(pos_df),), dtype=bool)
    pos_df['train_mask'][:int(0.8 * len(pos_list))] = True
    pos_df['train_mask'][int(0.8 * len(pos_list)):] = False
    pos_df['val_mask'][int(0.8 * len(pos_df)): int(0.9 * len(pos_df))] = True
    pos_df['val_mask'][:int(0.8 * len(pos_df))] = False
    pos_df['val_mask'][int(0.9 * len(pos_df)):] = False
    pos_df['test_mask'][:int(0.9 * len(pos_list))] = False
    pos_df['test_mask'][int(0.9 * len(pos_list)):len(pos_list)] = True
    train_indices = pos_df['train_mask'].to_numpy().nonzero()[0]
    val_indices = pos_df['val_mask'].to_numpy().nonzero()[0]
    test_indices = pos_df['test_mask'].to_numpy().nonzero()[0]

    # print(train_index)
    for s in trust_f:
        uid = s[0]
        fid = s[1]
        if uid > user_count or fid > user_count:
            continue
        trust_list.append([uid, fid])

    trust_df = pd.DataFrame(trust_list, columns=['uid', 'fid'])
    trust_df = trust_df.sort_values(axis=0, ascending=True, by='uid').reset_index()
    # trusted_df = trust_df.copy()
    # trusted_df['uid'] = trust_df['fid']  # trust使单向关系
    # trusted_df['fid'] = trust_df['uid']
    # trust_df = pd.concat([trust_df, trusted_df], axis=0).reset_index()
    trust_df -= 1
    print(min(trust_df['uid']))
    print(max(trust_df['uid']))

    """
    # 读mat文件
    dataFile1 = '../dataset/epinions_7411_8728/Epinions.mat'
    data1 = scio.loadmat(dataFile1)
    print(data1['rating'])
    data1r = pd.DataFrame(list(data1['rating']), columns=['uid', 'iid', 'label'])
    print(data1r)
    print(data1r['uid'].value_counts())
    """

    #####################################################################################################
    # cold-start dataset
    """
    cold_df = pos_df['uid'].value_counts()
    print(cold_df)
    cold_df = cold_df[cold_df <= 20]
    cold_df = cold_df[cold_df > 2]
    cold_start_users = cold_df.index
    co = pd.DataFrame()
    tr_co = pd.DataFrame()
    tred_co = pd.DataFrame()
    for co_user in cold_start_users:
        co = pd.concat([co, pos_df[pos_df.uid == co_user]], axis=0)
        tr_co = pd.concat([tr_co, trust_df[trust_df.uid == co_user]], axis=0)
        tred_co = pd.concat([tred_co, trust_df[trust_df.fid == co_user]], axis=0)

    tr_co = tr_co.drop(columns=['index'])
    tred_co = tred_co.drop(columns=['index'])
    print(tr_co)
    tr_co = tr_co.merge(tred_co, how='inner')
    print(tr_co)  # 社交关系最后user重新编码也要能对应上
    co['uid'] = co['uid'].astype('category')
    co['iid'] = co['iid'].astype('category')
    d_uid = dict(enumerate(co['uid'].cat.categories))  # 新旧id对应关系
    d_uid = dict(zip(d_uid.values(), d_uid.keys()))  # 交换key和val
    print(d_uid)
    co['uid'] = co['uid'].cat.codes.values + 1
    co['iid'] = co['iid'].cat.codes.values + 1
    co = co.reset_index()
    co = co.drop(columns=['index', 'train_mask', 'test_mask'])
    print(co)
    tr_co['uid'] = [d_uid[x]+1 for x in tr_co['uid']]
    tr_co['fid'] = [d_uid[x]+1 for x in tr_co['fid']]
    print(tr_co)
    mean_rating = np.mean(co['label'])
    print(mean_rating)
    print(max(co['iid']))
    co.to_csv(outputdir + '/cold_rating.csv', sep=' ', header=0, index=0)
    tr_co.to_csv(outputdir + '/cold_trust.csv', sep=' ', header=0, index=0)

    ##########
    
    user_df = pd.DataFrame([i for i in range(1, max(co['uid']) + 1)], columns=['uid'])
    item_df = pd.DataFrame([i for i in range(1, max(co['iid']) + 1)], columns=['iid'])
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_df, 'uid', 'uid')
    graph_builder.add_entities(item_df, 'iid', 'iid')
    graph_builder.add_binary_relations(co, 'uid', 'iid', 'rated')
    graph_builder.add_binary_relations(co, 'iid', 'uid', 'rated-by')
    graph_builder.add_trust_relations(tr_co, 'uid', 'fid', 'trust')

    g = graph_builder.build()
    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.

    g.edges['rated'].data['rating'] = torch.tensor(co['label'].values, dtype=torch.long)
    # g.edges['rated'].data['boun3'] = torch.tensor(co['boun3'].values, dtype=torch.long)
    # g.edges['rated'].data['boun4'] = torch.tensor(co['boun4'].values, dtype=torch.long)
    # g.edges['rated'].data['boun5'] = torch.tensor(co['boun5'].values, dtype=torch.long)
    # g.edges['rated'].data['boun1'] = torch.tensor(co['boun1'].values, dtype=torch.long)
    # g.edges['rated'].data['boun2'] = torch.tensor(co['boun2'].values, dtype=torch.long)
    g.edges['rated-by'].data['rating'] = torch.tensor(co['label'].values, dtype=torch.long)
    # g.edges['rated-by'].data['boun3'] = torch.tensor(co['boun3'].values, dtype=torch.long)
    # g.edges['rated-by'].data['boun4'] = torch.tensor(co['boun4'].values, dtype=torch.long)
    # g.edges['rated-by'].data['boun5'] = torch.tensor(co['boun5'].values, dtype=torch.long)
    # g.edges['rated-by'].data['boun1'] = torch.tensor(co['boun1'].values, dtype=torch.long)
    # g.edges['rated-by'].data['boun2'] = torch.tensor(co['boun2'].values, dtype=torch.long)
    if not no_helpfulness:
        g.edges['rated'].data['helpfulness'] = torch.tensor(co['helpfulness'].values, dtype=torch.long)
        g.edges['rated-by'].data['helpfulness'] = torch.tensor(co['helpfulness'].values, dtype=torch.long)

    indice_t = tr_co['uid'].index

    co['train_mask'] = np.ones((len(co),), dtype=bool)
    co['val_mask'] = np.ones((len(co),), dtype=bool)
    co['test_mask'] = np.ones((len(co),), dtype=bool)
    co['train_mask'][:int(0.7 * len(co))] = True
    co['train_mask'][int(0.7 * len(co)):] = False
    co['val_mask'][int(0.7 * len(co)): int(0.85 * len(co))] = True
    co['val_mask'][:int(0.7 * len(co))] = False
    co['val_mask'][int(0.85 * len(co)):] = False
    co['test_mask'][:int(0.85 * len(co))] = False
    co['test_mask'][int(0.85 * len(co)):len(co)] = True
    train_indices = co['train_mask'].to_numpy().nonzero()[0]
    val_indices = co['val_mask'].to_numpy().nonzero()[0]
    test_indices = co['test_mask'].to_numpy().nonzero()[0]

    train_g = build_train_graph(g, train_indices, indice_t, 'rated', 'rated-by', 'trust')
    val_g = build_train_graph(g, val_indices, indice_t, 'rated', 'rated-by', 'trust')
    test_g = build_train_graph(g, test_indices, indice_t, 'rated', 'rated-by', 'trust')

    mean_rating = np.mean(co['label'])
    with open(outputdir + '/colddataset.pkl', 'wb') as f:
        pickle.dump(co, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tr_co, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(d_uid, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_rating, f, pickle.HIGHEST_PROTOCOL)
    ##################################################################################################
    """

    # for normal users 和冷启动的数据处理分开执行

    user_df = pd.DataFrame([i for i in range(0, user_count)], columns=['uid'])  # id 和边从0开始吗
    item_df = pd.DataFrame([i for i in range(0, item_count)], columns=['iid'])  # id 和边从0开始吗
    print(item_df)
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_df, 'uid', 'uid')
    graph_builder.add_entities(item_df, 'iid', 'iid')
    graph_builder.add_binary_relations(pos_df, 'uid', 'iid', 'rated')
    graph_builder.add_binary_relations(pos_df, 'iid', 'uid', 'rated-by')
    graph_builder.add_trust_relations(trust_df, 'uid', 'fid', 'trust')
    # graph_builder.add_trust_relations(trusted_df, 'uid', 'fid', 'trust')

    g = graph_builder.build()
    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.edges['rated'].data['rating'] = torch.tensor(pos_df['label'].values, dtype=torch.long)
    g.edges['rated-by'].data['rating'] = torch.tensor(pos_df['label'].values, dtype=torch.long)
    print(g.edata)

    """
    
    g.edges['rated'].data['boun3'] = torch.tensor(pos_df['boun3'].values, dtype=torch.long)
    g.edges['rated'].data['boun4'] = torch.tensor(pos_df['boun4'].values, dtype=torch.long)
    g.edges['rated'].data['boun5'] = torch.tensor(pos_df['boun5'].values, dtype=torch.long)
    g.edges['rated'].data['boun1'] = torch.tensor(pos_df['boun1'].values, dtype=torch.long)
    g.edges['rated'].data['boun2'] = torch.tensor(pos_df['boun2'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun3'] = torch.tensor(pos_df['boun3'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun4'] = torch.tensor(pos_df['boun4'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun5'] = torch.tensor(pos_df['boun5'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun1'] = torch.tensor(pos_df['boun1'].values, dtype=torch.long)
    g.edges['rated-by'].data['boun2'] = torch.tensor(pos_df['boun2'].values, dtype=torch.long)
    
    """

    if not no_helpfulness:
        g.edges['rated'].data['helpfulness'] = torch.tensor(pos_df['helpfulness'].values, dtype=torch.long)
        g.edges['rated-by'].data['helpfulness'] = torch.tensor(pos_df['helpfulness'].values, dtype=torch.long)

    indice_t = trust_df['uid'].index
    # Build the graph with training interactions only.
    # print(indice_t)
    # print(g.edata)

    train_g = build_train_graph(g, train_indices, indice_t, 'rated', 'rated-by', 'trust')
    print(train_g)
    print(g)
    test_g = build_train_graph(g, test_indices, indice_t, 'rated', 'rated-by', 'trust')
    val_g = build_train_graph(g, val_indices, indice_t, 'rated', 'rated-by', 'trust')

    # test_matrix = build_test_matrix(g, test_indices, 'uid', 'iid', 'rated')
    # print(train_g)
    # print(test_matrix)
    mean_rating = np.mean(pos_df['label'])

    with open(outputdir + '/dataset.pkl', 'wb') as f:
        pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trust_df, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val_g, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_rating, f, pickle.HIGHEST_PROTOCOL)


    neg_candidates = []
    counter = 0
    for i in range(user_count):
        for j in range(item_count):
            if counter < len(pos_df):
                if i == pos_df.iloc[counter, 0] and j == pos_df.iloc[counter, 1]:
                    counter += 1
                else:
                    neg_candidates.append([i, j])
            else:
                neg_candidates.append([i, j])
    neg_candidates = np.array(neg_candidates)

    idx = np.random.choice(len(neg_candidates), len(val_indices) + len(test_indices), replace=False)
    val_neg_candidates = neg_candidates[sorted(idx[:len(val_indices)])]
    test_neg_candidates = neg_candidates[sorted(idx[len(val_indices):])]

    train_neg_candidates = []
    counter = 0
    for i in range(user_count):
        for j in range(item_count):
            if counter < len(train_df):
                if i == train_df.iloc[counter, 0] and j == train_df.iloc[counter, 1]:
                    counter += 1
                else:
                    train_neg_candidates.append([i, j])
            else:
                train_neg_candidates.append([i, j])
    train_neg_candidates = np.array(train_neg_candidates)

    np.savez(outputdir + '/train_val_test_neg_user_artist.npz',
             train_neg_user_artist=train_neg_candidates,
             val_neg_user_artist=val_neg_candidates,
             test_neg_user_artist=test_neg_candidates)