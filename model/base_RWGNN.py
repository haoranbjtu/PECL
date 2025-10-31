import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class Path_encoder(nn.Module):  # 序列encoder
    def __init__(self,
                 origin_g,
                 etypes,
                 out_dim,
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=True,
                 attn_switch=True):
        super(Path_encoder, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.rnn_type = rnn_type
        self.r_vec = r_vec  # edge emb
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch
        self.etypes = etypes  # ?
        self.rating_rotation = True
        self.rel_linear = nn.Linear(out_dim, out_dim)
        self.rel_l2 = nn.Linear(out_dim*2, out_dim)
        nn.init.xavier_normal_(self.rel_linear.weight, gain=1.414)
        nn.init.xavier_normal_(self.rel_l2.weight, gain=1.414)

        # rnn-like RWpath instance aggregator
        # consider multiple attention heads
        if rnn_type == 'gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)

        # node-level attention
        # attention considers the center node embedding or not
        if self.attn_switch:
            self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x


        # weight initialization
        if self.attn_switch:
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        else:
            nn.init.xavier_normal_(self.attn.data, gain=1.414)

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges):
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, rating_indices, target_idx = inputs
        else:
            g, features, type_mask, edge_metapath_indices, rating_indices = inputs

        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)  # take nodes' emb from feature tensor(like a dic)

        # 生成时间嵌入矩阵
        emb = torch.zeros(*rating_indices.size(), self.out_dim, device=edata.device)
        for i in range(self.out_dim):
            if i % 2 == 0:
                emb[:, :, i] = torch.sin(rating_indices / 10000 ** (2 * i / self.out_dim))
            else:
                emb[:, :, i] = torch.cos(rating_indices / 10000 ** (2 * (i - 1) / self.out_dim))


        # rating_emb = F.embedding(rating_indices, torch.cat((self.r_vec[:, :, 0], self.r_vec[:, :, 1]), 1))  # 这里-1 遇到 u-u 就变成-1 就报错
        # rating_emb = self.rel_linear(emb)  # rating_emb也更新
        rating_emb = emb.reshape(edata.shape[0], rating_indices.shape[1], -1, 2)  # chunk cat的反向操作  报错

        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':  # 改成用评分emb
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)

        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(rating_emb, p=2, dim=2)  # for rotation, make the rating emb unit tensor
            # r_vec = rating_emb  # n_path, path_len-1, out_dim/2, 2
            # p=2 二范数
            # 初始化边嵌入
            if self.rnn_type == 'RotatE0':  # R0 num_edge_type/2, in_dim/2, 2
                r_vec = torch.stack((r_vec, r_vec), dim=1)  # num_edge_type/2, 2, in_dim/2, 2
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]  # 虚部取反 表示逆关系
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
            # n_path, path_len, out_dim/2, 2
            final_r_vec = torch.zeros([edata.shape[0], edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            # path_len, out_dim/2, 2
            final_r_vec[:, -1, :, 0] = 1  # real part initialized as 1
            # print(self.etypes)  # rating, social 这样的边类型
            if self.rating_rotation:  # haory
                """
                for i in range(final_r_vec.shape[1] - 2, -1, -1):  # 目前 1分是0度  u-u也是0度，即不旋转
                    print(r_vec[:, i, :, 0])
                    final_r_vec[:, i, :, 0] = final_r_vec[:, i + 1, :, 0].clone() * r_vec[:, i, :, 0] - \
                                              final_r_vec[:, i + 1, :, 1].clone() * r_vec[:, i, :, 1]  # 实部
                    final_r_vec[:, i, :, 1] = final_r_vec[:, i + 1, :, 0].clone() * r_vec[:, i, :, 1] + \
                                              final_r_vec[:, i + 1, :, 1].clone() * r_vec[:, i, :, 0]  # imaginary
                """
                for i in range(edata.shape[1] - 1):  # 路径终点被忽略  在GAT中计算
                    temp1 = edata[:, i, :, 0].clone() * r_vec[:, i, :, 0] - \
                            edata[:, i, :, 1].clone() * r_vec[:, i, :, 1]  # 实部
                    temp2 = edata[:, i, :, 0].clone() * r_vec[:, i, :, 1] + \
                            edata[:, i, :, 1].clone() * r_vec[:, i, :, 0]
                    edata[:, i, :, 0] = temp1
                    edata[:, i, :, 1] = temp2
                edata = edata.reshape(edata.shape[0], edata.shape[1], -1)  # n_path, path_len, out_emb
                hidden = torch.mean(edata, dim=1)  # n_path, out_emb
                hidden = torch.cat([hidden] * self.num_heads, dim=1)  # n_path, out_emb*n_heads
                hidden = hidden.unsqueeze(dim=0)  # 1, n_path, out_emb*n_heads
            else:
                for i in range(final_r_vec.shape[0] - 2, -1, -1):  # 路径长度
                    # consider None edge (symmetric relation)  self.etypes[i]换成
                    if self.etypes[i] is not None:
                        final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                               final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]  # 实部
                        final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                               final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :,
                                                                                  0]  # imaginary
                    else:  # 如果是None 就不旋转
                        final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                        final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
                for i in range(edata.shape[1] - 1):  # 路径终点被忽略  在GAT中计算
                    temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                            edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]  # 实部
                    temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                            edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                    edata[:, i, :, 0] = temp1
                    edata[:, i, :, 1] = temp2
                edata = edata.reshape(edata.shape[0], edata.shape[1], -1)  # n_path, path_len, out_emb
                hidden = torch.mean(edata, dim=1)
                hidden = torch.cat([hidden] * self.num_heads, dim=1)
                hidden = hidden.unsqueeze(dim=0)

        elif self.rnn_type == 'neighbor':  # 元路径邻居
            hidden = edata[:, 0]
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim 路径emb
        # if self.attn_switch:  # attention considers the center node embedding or not
        center_node_feat = F.embedding(edge_metapath_indices[:, -1], features)  # E x out_dim
        a1 = self.attn1(center_node_feat)  # E x num_heads
        a2 = (eft * self.attn2).sum(dim=-1)  # E x num_heads
        a = (a1 + a2).unsqueeze(dim=-1)  # E x num_heads x 1

        # bi_node_feat = F.embedding(edge_metapath_indices[:, -2], features).repeat(1, self.num_heads).view(eft.shape[0], self.num_heads, self.out_dim)
        # eft = self.rel_l2(torch.cat((bi_node_feat, eft), 2))  # 拼接
        """
        eft = eft.reshape(eft.shape[0], eft.shape[1], eft.shape[2] // 2, 2)
        ef = torch.zeros_like(eft, device=eft.device)
        bi_node_feat = F.normalize(bi_node_feat, p=2, dim=2)  # 归一化
        bi_node_feat = bi_node_feat.reshape(bi_node_feat.shape[0], bi_node_feat.shape[1], bi_node_feat.shape[2] // 2, 2)
        temp1 = eft[:, :, :, 0].clone() * bi_node_feat[:, :, :, 0] - \
                eft[:, :, :, 1].clone() * bi_node_feat[:, :, :, 1]  # 实部
        temp2 = eft[:, :, :, 0].clone() * bi_node_feat[:, :, :, 1] + \
                eft[:, :, :, 1].clone() * bi_node_feat[:, :, :, 0]
        ef[:, :, :, 0] = temp1
        ef[:, :, :, 1] = temp2

        ef = ef.reshape(eft.shape[0], eft.shape[1], -1)  # n_path, path_len, out_emb
        """
        # else:
            # a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # E x num_heads x 1
        a = self.leaky_relu(a)  # 低阶信息与高阶信息的交互  旋转 mean cat
        g.edata.update({'eft': eft, 'a': a})  # 消息传递 聚合 路径emb   计算图 由采样路径构成 user-user 是同构图  +center_node_feat/2, eft换成节点emb(传统gat) 看路径encoder是否有意义
        # compute softmax normalized attention values
        self.edge_softmax(g)
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(self.message_passing, fn.sum('ft', 'ft'))  #
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if self.use_minibatch:
            return ret[target_idx]  # 一个batch输入节点的emb
        else:
            return ret


class Paths_att_aggre(nn.Module):  # 聚合metapath instance
    def __init__(self,
                 num_metapaths,
                 origin_g,
                 etypes_list,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 use_minibatch=False):
        super(Paths_att_aggre, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.metapath_layers.append(Path_encoder(
                origin_g,
                etypes_list[i],
                out_dim,
                num_heads,
                rnn_type,
                r_vec,
                attn_drop=attn_drop,
                use_minibatch=use_minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)  # * num_heads
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:  # 默认分batch
            g_list, features, type_mask, edge_metapath_indices_list, rating_indices_list, target_idx_list = inputs
            # metapath-specific layers
            metapath_outs = [F.elu(
                metapath_layer((g, features, type_mask, edge_metapath_indices, rating_indices, target_idx)).view(-1,
                                                                                                     self.num_heads * self.out_dim))
                for g, edge_metapath_indices, rating_indices, target_idx, metapath_layer in
                zip(g_list, edge_metapath_indices_list, rating_indices_list, target_idx_list, self.metapath_layers)]
        else:
            g_list, features, type_mask, edge_metapath_indices_list, rating_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(
                metapath_layer((g, features, type_mask, edge_metapath_indices, rating_indices)).view(-1,
                                                                                                     self.num_heads * self.out_dim))
                for g, edge_metapath_indices, rating_indices, metapath_layer in
                zip(g_list, edge_metapath_indices_list, rating_indices_list, self.metapath_layers)]  # 输入格式不一样  没有g_list

        beta = []
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)   # 有时候beta为空就会报错 (val)   调大batch_size会  数都太小 导致经过softmax后区分不开 ciao 乘1000 epinions乘100
        # su = torch.sum(beta)
        beta = F.softmax(beta, dim=0)
        # beta = beta / su
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)
        return h
