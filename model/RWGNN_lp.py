import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv

from model.base_RWGNN import Paths_att_aggre


# mini-batch

# for link prediction task

class RWGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,  # 边种类 需要改成 评分数  记得还原
                 origin_g,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(RWGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.origin_g = origin_g

        # etype-specific parameters  边嵌入 定义之处
        """
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':  # 只改了这个
            one_vec = nn.Parameter(torch.ones(size=(num_edge_type, in_dim // 2, 2)))  # 边的种类 和评分种类 不一样
        # if r_vec is not None:
        #    nn.init.xavier_normal_(r_vec.data, gain=1.414)
        # edge初始化 u-i 用 评分角度初始化
        """
        # one_vec = nn.Parameter(torch.ones(size=(num_edge_type, in_dim // 2, 2)))  # 边的种类 和评分种类 不一样
        # r_vec = nn.Embedding(5, in_dim // 2)
        r_vec = torch.ones(size=(2, in_dim // 2, 2))
        # angles = torch.linspace(0, math.pi/2, 2)  # 从-pi/2到 pi/2  区分没评分（3分）和低评分
        # r_vec_init[:, :, 0] = torch.cos(angles).unsqueeze(1).repeat(1, in_dim // 2)
        # r_vec_init[:, :, 1] = torch.sin(angles).unsqueeze(1).repeat(1, in_dim // 2)
        # r_vec = nn.Parameter(r_vec_init)

        # ctr_ntype-specific layers
        self.user_layer = Paths_att_aggre(num_metapaths_list[0],
                                          origin_g,
                                          etypes_lists[0],
                                          in_dim,
                                          num_heads,
                                          attn_vec_dim,
                                          rnn_type,
                                          r_vec,
                                          attn_drop,
                                          use_minibatch=True)
        self.item_layer = Paths_att_aggre(num_metapaths_list[1],
                                          origin_g,
                                          etypes_lists[1],
                                          in_dim,
                                          num_heads,
                                          attn_vec_dim,
                                          rnn_type,
                                          r_vec,
                                          attn_drop,
                                          use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_user = nn.Linear(in_dim*2, out_dim, bias=True)
        self.fc_item = nn.Linear(in_dim*2, out_dim, bias=True)
        self.fc_u = nn.Linear(in_dim*num_heads, out_dim, bias=True)
        self.fc_i = nn.Linear(in_dim*num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)  # 线性层参数初始化
        nn.init.xavier_normal_(self.fc_item.weight, gain=1.414)
        self.rp1 = nn.Linear(out_dim*2, out_dim, bias=True)
        self.rp2 = nn.Linear(out_dim, 8)
        self.rp3 = nn.Linear(8, 1)
        nn.init.xavier_normal_(self.rp1.weight, gain=1.414)  # 线性层参数初始化
        nn.init.xavier_normal_(self.rp2.weight, gain=1.414)
        nn.init.xavier_normal_(self.rp3.weight, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(0.1)
        # self.gat_item = GATConv((in_dim, in_dim), in_dim, num_heads, allow_zero_in_degree=True)
        # self.gat_user = GATConv((in_dim, in_dim), in_dim, num_heads, allow_zero_in_degree=True)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, rating_indices_lists, target_idx_lists, (items_batch, users_batch), num_items = inputs

        # ctr_ntype-specific layers
        h_user = self.user_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], rating_indices_lists[0], target_idx_lists[0]))  # user item位置可能需要互换
        h_item = self.item_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], rating_indices_lists[1], target_idx_lists[1]))
        # 二部图
        """
        user_emb = F.embedding(users_batch+num_items, features)
        item_emb = F.embedding(items_batch, features)
        # subgraph = dgl.node_subgraph(self.origin_g, {'uid': users_batch, 'iid': items_batch})
        user_sub_g = dgl.sampling.sample_neighbors(self.origin_g.edge_type_subgraph(['rated'], {'uid': users_batch}, 6)
        item_neigh = user_sub_g.nodes()
        item_sub_g = dgl.sampling.sample_neighbors(self.origin_g.edge_type_subgraph(['rated-by'], {'iid': items_batch}, 6)
        bipartite_item_emb = (self.gat_item(item_sub_g, (user_emb, item_emb))).reshape(user_emb.shape[0], -1) # 图太小 没什么邻居 反而效果更差
        bipartite_user_emb = (self.gat_user(user_sub_g, (item_emb, user_emb))).reshape(user_emb.shape[0], -1)

        # 拼接 二部图 与 元路径
        h_user = torch.cat([h_user, bipartite_user_emb], 1)
        h_item = torch.cat([h_item, bipartite_item_emb], 1)
        """
        user_emb = F.embedding(users_batch, features)
        item_emb = F.embedding(items_batch, features)
        h_user = self.fc_u(h_user)
        h_item = self.fc_i(h_item)
        logits_user = self.fc_user(torch.concat((h_user, user_emb), dim=1))
        logits_item = self.fc_item(torch.concat((h_item, item_emb), dim=1))
        # logits_user = self.fc_user(h_user)
        # logits_item = self.fc_item(h_item)

        #user_emb = self.fc_u(user_emb)
        #item_emb = self.fc_i(item_emb)
        # logits_user = h_user
        # logits_item = h_item
        rating = self.rp3(self.rp2(self.rp1(torch.concat((logits_item, logits_user), dim=1))))  # 评分预测公式
        # rating = 5 * torch.sigmoid(rating)  # 试试去掉  去掉效果变差

        # embedding_item = logits_user.view(-1, 1, logits_item.shape[1])  # 因为这次id设置是item user 所以和程序相反
        # embedding_user = logits_item.view(-1, logits_item.shape[1], 1)
        # rating = torch.bmm(embedding_item, embedding_user)  # 评分公式1 内积  # rating_label需要整成迭代器
        return [logits_user, logits_item], [h_user, h_item], rating


class RWGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 origin_g,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(RWGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after transformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # RWGNN_lp layers
        self.layer1 = RWGNN_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     origin_g,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, rating_indices_lists, target_idx_lists, (items_batch, users_batch) = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])  # 这里可以用张量替代循环
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_user, logits_item], [h_user, h_item], rating = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, rating_indices_lists, target_idx_lists, (items_batch, users_batch), features_list[0].shape[0]))

        return [logits_user, logits_item], [h_user, h_item], rating, transformed_features
