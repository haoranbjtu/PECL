#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch深度学习和图神经网络(卷2）——开发应用>配套代码
@配套代码技术支持：bbs.aianaconda.com
"""
import os
import re
import pickle
import pandas as pd
import dask.dataframe as dd

import torch
import dgl
import numpy as np
import scipy.sparse as ssp
from movielens_util.PandasGraphBuilder import PandasGraphBuilder


def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = dgl.edge_subgraph(g,
        {etype: train_indices, etype_rev: train_indices},
        preserve_nodes=True)

    # remove the induced node IDs - should be assigned by model instead
    del train_g.nodes[utype].data[dgl.NID]
    del train_g.nodes[itype].data[dgl.NID]

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            # print(col)
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID].type(torch.long)]

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
    print(len(val_src))
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix


def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / \
           (values.max(0, keepdims=True) - values.min(0, keepdims=True))


# 最流行的ml数据集划分方式


def train_test_split_by_time(df, timestamp, user):
    df['train_mask'] = np.ones((len(df),), dtype=bool)
    df['val_mask'] = np.zeros((len(df),), dtype=bool)
    df['test_mask'] = np.zeros((len(df),), dtype=bool)
    df = dd.from_pandas(df, npartitions=1)

    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    df = df.groupby(user, group_keys=False).apply(train_test_split).compute(
        scheduler='processes').sort_index()  # warning , meta=
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return df['train_mask'].to_numpy().nonzero()[0], \
           df['val_mask'].to_numpy().nonzero()[0], \
           df['test_mask'].to_numpy().nonzero()[0]


if __name__ == '__main__':
    directory = '../data/raw data/ml-1m'
    output_path = '../data/ml_1m.pkl'

    # ml-1m的处理部分直接用PinSAGE的

    # Load data
    users = []
    with open(os.path.join(directory, 'users.dat'), encoding='latin1') as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split('::')
            users.append({'user_id': int(id_),
                          'gender': gender,
                          'age': age,
                          'occupation': occupation,
                          'zip': zip_})
    users = pd.DataFrame(users).astype('category')  # 只用了id
    # print(users)

    movies = []
    with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
        for l in f:
            id_, title, genres = l.strip().split('::')
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*\([0-9]{4}\)$', title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {'movie_id': int(id_), 'title': title, 'year': year}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = pd.DataFrame(movies).astype({'year': 'category'})
    # pd.set_option('display.max_columns', None)  # 展示所有列
    # print(movies.head())

    ratings = []
    with open(os.path.join(directory, 'ratings.dat'), encoding='latin1') as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,  # 新加的
                'timestamp': timestamp,
            })
    ratings = pd.DataFrame(ratings)
    # print(ratings)

    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = ratings['user_id'].unique()
    distinct_movies_in_ratings = ratings['movie_id'].unique()
    users = users[users['user_id'].isin(distinct_users_in_ratings)]
    movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]

    # Group the movie features into genres (a vector), year (a category), title (a string)
    genre_columns = movies.columns.drop(['movie_id', 'title', 'year'])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')  # 将电影类型加工成bool类型
    movies_categorical = movies.drop('title', axis=1)  # 去掉标题列

    # Build heterogeneous graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')  # 加节点
    graph_builder.add_entities(movies_categorical, 'movie_id', 'movie')
    graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'rated')  # 加边
    graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'rated-by')

    g = graph_builder.build()

    # graph_builder.edges_per_relation  # 边关系
    # graph_builder.num_nodes_per_type  # 节点数

    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    # 转成张量数组

    # 看需要是否放进图内 可能占内存
    g.nodes['user'].data['gender'] = torch.LongTensor(users['gender'].cat.codes.values)  # warning
    g.nodes['user'].data['age'] = torch.LongTensor(users['age'].cat.codes.values)
    g.nodes['user'].data['occupation'] = torch.LongTensor(users['occupation'].cat.codes.values)
    g.nodes['user'].data['zip'] = torch.LongTensor(users['zip'].cat.codes.values)

    g.nodes['movie'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
    # 转为索引向量Categories (81, object): [1919, 1920, 1921, 1922, ..., 1997, 1998, 1999, 2000]
    g.nodes['movie'].data['genre'] = torch.FloatTensor(movies[genre_columns].values)

    g.edges['rated'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    g.edges['rated'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['rated-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['rated-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)

    # Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')

    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'user', 'movie', 'rated', 'rated-by')

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'rated')

    print(train_g)
    print(val_matrix.shape)
    print(val_matrix)
    # Build title set
    movie_textual_dataset = {'title': movies['title'].values}

    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': movie_textual_dataset,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
