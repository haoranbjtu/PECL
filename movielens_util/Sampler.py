# -*- coding: utf-8 -*-
import pickle
import dgl
import torch
import numpy as np
import random


"""
Sampler for Bipartite graph
1 sampler of Node2vec (p, q to control the random walk)  -- Homogeneous graph
2 meta-path random walk   -- Heterogeneous graph
3 
"""


class base_Sampler():
    def __init__(self):
        super(base_Sampler).__init__()

    def sample(self, g, dataset):
        pass


class metapath_RW_Sampler(base_Sampler):
    def __init__(self):
        super(metapath_RW_Sampler).__init__()

    def sample(self, g, dataset):
        # mini batch 为user item单独取seeds
        # 得研究dataloader collator
        """
        每个seed采样多条路径  for循环 然后拼接整理路径？
        先实现不分batch版本 之后再改
        ml-1m  movie 3705  user  6039 大小不用分batch
        负采样
        :param dataset:
        :return:
        """
        # print(g.ndata)
        # print(g.ndata['id']['movie'])
        # print(g.ndata['id']['user'])
        # print(dgl.sampling.random_walk(g, g.ndata['id']['movie'], metapath=['rated-by', 'rated']*2)[0])
        # 5 nodes per route, 4 routes per seeds
        movie_path = dgl.sampling.random_walk(g, g.ndata['id']['movie'], metapath=['rated-by', 'rated']*2)[0]
        user_path = dgl.sampling.random_walk(g, g.ndata['id']['user'], metapath=['rated', 'rated-by']*2)[0]

        for i in range(3):
            movie_path = torch.cat((movie_path, dgl.sampling.random_walk(g,
                                                                         g.ndata['id']['movie'],
                                                                         metapath=['rated-by', 'rated']*2)[0]),
                                   dim=0)
            user_path = torch.cat((user_path, dgl.sampling.random_walk(g,
                                                                       g.ndata['id']['user'],
                                                                       metapath=['rated', 'rated-by']*2)[0]),
                                  dim=0)

        # print(movie_path.shape)
        return movie_path, user_path


if __name__ == '__main__':
    dataset_path = '../data/ml_1m.pkl'
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    g = dataset['train-graph']
    item_texts = dataset['item-texts']

    # 设置节点属性值
    g.nodes['user'].data['id'] = torch.arange(g.number_of_nodes('user')).type(torch.int32)
    g.nodes['movie'].data['id'] = torch.arange(g.number_of_nodes('movie')).type(torch.int32)
    metapath_RW_Sampler = metapath_RW_Sampler()
    metapath_RW_Sampler.sample(g, 'ml_1m')