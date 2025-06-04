import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import faiss
# paper: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW'22
from data.augmentor import GraphAugmentor


class NCL(GraphRecommender):
    # def __init__(self, conf, training_set, test_set, training_time, test_time):
    #     super(NCL, self).__init__(conf, training_set, test_set, training_time, test_time)
    def __init__(self, conf, training_set, test_set):
        super(NCL, self).__init__(conf, training_set, test_set)
        args = self.config['NCL']
        self.n_layers = int(args['n_layer'])
        self.ssl_temp = float(args['tau'])
        self.ssl_reg = float(args['ssl_reg'])
        # self.time_reg = float(args['time_reg'])
        self.path_reg = float(args['path_reg'])
        self.multi_reg = float(args['multi_reg'])
        self.cl_rate = float(args['lambda'])
        self.hyper_layers = int(args['hyper_layers'])
        self.alpha = float(args['alpha'])
        self.proto_reg = float(args['proto_reg'])
        self.k = int(args['num_clusters'])
        self.temp = float(args['temp'])
        self.aug_type = int(args['aug_type'])
        self.drop_rate = float(args['drop_rate'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.temp, self.aug_type, self.drop_rate)
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

    def e_step(self):
        user_embeddings = self.model.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict['item_emb'].detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]
        proto_nce_loss_user = InfoNCE(user_emb[user_idx],user2centroids,self.ssl_temp) * self.batch_size
        item2cluster = self.item_2cluster[item_idx]
        item2centroids = self.item_centroids[item2cluster]
        proto_nce_loss_item = InfoNCE(item_emb[item_idx],item2centroids,self.ssl_temp) * self.batch_size
        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss
    
    # def ssl_time_loss(self, time_emb, initial_emb, user, item):
    #     time_user_emb_all, time_item_emb_all = torch.split(time_emb, [self.data.user_num, self.data.item_num])
    #     initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
    #     time_user_emb = time_user_emb_all[user]
    #     initial_user_emb = initial_user_emb_all[user]
    #     norm_user_emb1 = F.normalize(time_user_emb)
    #     norm_user_emb2 = F.normalize(initial_user_emb)
    #     norm_all_user_emb = F.normalize(initial_user_emb_all)
    #     pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
    #     ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
    #     pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
    #     ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
    #     ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

    #     time_item_emb = time_item_emb_all[item]
    #     initial_item_emb = initial_item_emb_all[item]
    #     norm_item_emb1 = F.normalize(time_item_emb)
    #     norm_item_emb2 = F.normalize(initial_item_emb)
    #     norm_all_item_emb = F.normalize(initial_item_emb_all)
    #     pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
    #     ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
    #     pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
    #     ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
    #     ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

    #     ssl_loss = self.time_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
    #     return ssl_loss

    def ssl_path_loss(self, path_emb, initial_emb, user, item):
        path_user_emb_all, path_item_emb_all = torch.split(path_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        path_user_emb = path_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(path_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        path_item_emb = path_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(path_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.path_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss
    
    def ssl_multi_loss(self, multi_emb, initial_emb, user, item):
        multi_user_emb_all, multi_item_emb_all = torch.split(multi_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        multi_user_emb = multi_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(multi_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        multi_item_emb = multi_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(multi_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.multi_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            # dropped_adj1 = model.graph_reconstruction()
            # dropped_adj2 = model.graph_reconstruction()
            if epoch >= 20:
                self.e_step()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                model.train()
                # rec_user_emb, rec_item_emb, emb_list = model()
                rec_user_emb, rec_item_emb, emb_list, path_list  = model()
                # rec_user_emb, rec_item_emb, emb_list, rate_list, time_list, path_list  = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                multi_emb  = emb_list[self.hyper_layers*2-1]
                
                ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,user_idx,pos_idx)
                multi_loss = self.ssl_multi_loss(context_emb,multi_emb,user_idx,pos_idx)
                
                # warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)/self.batch_size  + ssl_loss 
                warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)/self.batch_size  + ssl_loss + multi_loss

                if epoch<20: #warm_up
                    optimizer.zero_grad()
                    warm_up_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'multi_loss', multi_loss.item())
                else:
                    # Backward and optimize
                    
                    proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                    # rate_emb = rate_list[self.hyper_layers*2]
                    # time_emb = time_list[self.hyper_layers*2]
                    path_emb = path_list[self.hyper_layers*2]
                    # rate_loss = self.ssl_time_loss(context_emb,rate_emb,user_idx,pos_idx)
                    # time_loss = self.ssl_time_loss(context_emb,time_emb,user_idx,pos_idx)
                    path_loss = self.ssl_path_loss(context_emb,path_emb,user_idx,pos_idx)
                    
                    # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size + ssl_loss + multi_loss + proto_loss
                    # batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size + ssl_loss + multi_loss + time_loss + path_loss + proto_loss + cl_loss
                    batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size + ssl_loss + multi_loss + path_loss + proto_loss

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'multi_loss', multi_loss.item(), 'proto_loss', proto_loss.item())
                        # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'time_loss', time_loss.item(), 'path_loss', path_loss.item(), 'multi_loss', multi_loss.item(), 'proto_loss', proto_loss.item(), 'cl_loss', cl_loss.item())
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'path_loss', path_loss.item(), 'multi_loss', multi_loss.item(), 'proto_loss', proto_loss.item())

            model.eval()
            with torch.no_grad():
                # self.user_emb, self.item_emb, _ = model()# 进入前向传播函数
                self.user_emb, self.item_emb, _, path = model()# 进入前向传播函数
                # self.user_emb, self.item_emb, _, rate, time, path = model()# 进入前向传播函数
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            # self.best_user_emb, self.best_item_emb, _ = self.model()
            self.best_user_emb, self.best_item_emb, _, path = self.model()
            # self.best_user_emb, self.best_item_emb, _, rate, time, path = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, temp, aug_type, drop_rate):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.embedding_dict = self._init_model()
        self.norm_adj = data.norm_adj
        self.norm_rate = data.norm_rate
        # self.norm_time = data.norm_time
        self.norm_random_adj = data.norm_random_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        # self.sparse_norm_rate = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_rate).cuda()
        # self.sparse_norm_time = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_time).cuda()
        self.sparse_norm_random_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_random_adj).cuda()
        self.temp = temp
        self.aug_type = aug_type
        self.drop_rate = drop_rate

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict
    
    def graph_reconstruction(self):
        if self.aug_type == 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj
    
    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()
    
    def forward(self):
        # Initialize embeddings
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        # rate_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        # time_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        # path_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        # Track all embeddings across layers
        all_embeddings = [ego_embeddings]
        # all_rate_embeddings = [ego_embeddings]
        # all_time_embeddings = [ego_embeddings]
        all_path_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings1 = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            # ego_embeddings2 = torch.sparse.mm(self.sparse_norm_rate, ego_embeddings)
            # ego_embeddings3 = torch.sparse.mm(self.sparse_norm_time, ego_embeddings)
            ego_embeddings4 = torch.sparse.mm(self.sparse_norm_random_adj, ego_embeddings)
            
            
            # Update embeddings by adding path contributions with weights
            # ego_embeddings = 0.5*ego_embeddings1 + 0.5*ego_embeddings2
            # ego_embeddings = ego_embeddings1
            # rate_embeddings = ego_embeddings2
            # time_embeddings = ego_embeddings3
            # path_embeddings = ego_embeddings4

            all_embeddings.append(ego_embeddings1)
            # all_rate_embeddings.append(ego_embeddings2)
            # all_time_embeddings.append(ego_embeddings3)
            all_path_embeddings.append(ego_embeddings4)

        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        lgcn_all_embeddings = torch.mean(lgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lgcn_all_embeddings, [self.data.user_num, self.data.item_num])

        # user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        # item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]
        # return user_all_embeddings, item_all_embeddings, all_embeddings
        # return user_all_embeddings, item_all_embeddings, all_embeddings, all_rate_embeddings, all_time_embeddings, all_path_embeddings
        return user_all_embeddings, item_all_embeddings, all_embeddings, all_path_embeddings

    
    def forward2(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
    
    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward2(perturbed_mat1)
        user_view_2, item_view_2 = self.forward2(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1,view2,self.temp)
