import torch
import dgl
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

EPS = 1e-10


def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std


def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        print(row)
        print(indices)
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)  # 不适用于lastfm

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        # g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        # result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g)
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list


def parse_adjlist_LastFM(adjlist, edge_metapath_indices, rating_metapath_indices, samples=None, exclude=None,
                         offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    result_rating_indices = []
    for row, indices, rating_indices in zip(adjlist, edge_metapath_indices, rating_metapath_indices):
        # row包含每个节点的路径邻居，indices包含这个节点的所有某一元路径的路径
        # adjlist中的邻居节点是无偏置的，即从0开始，indices里的路径因为有两种节点所以有偏置
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:  # 1个以上邻居
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                    result_rating_indices.append(rating_indices[mask])
                else:
                    neighbors = row_parsed[1:]  # 邻居 这里邻居定义是路径终点
                    result_indices.append(indices)
                    result_rating_indices.append(rating_indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count  # 下采样 3/4 幂 常用
                p = np.array(p)
                p = p / p.sum()  # 采样概率

                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))  # 采样数
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                    result_rating_indices.append(rating_indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
                    print(indices)
                    print(sampled_idx)
                    result_rating_indices.append(rating_indices[sampled_idx])  # 报错
        else:  # ???
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            rating_indices = np.array([[0] * indices.shape[1]])  # ????
            if mode == 1:
                indices += offset
            result_indices.append(indices)
            result_rating_indices.append(rating_indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
            # 问题 只有u-i 有评分  u-u无 edges只有u-u对 i-i对
            # 计划用评分的地方是路径内  u-i-u这样  再思考一下

    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}  # 根据本次采样节点出现顺序重新编号
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))  # map 对edges所有元素使用lambda匿名函数，tup输入
    # (mapping[tup[0]], mapping[tup[1]]) 输出  即将edges的id都映射成本次采样生成的新id(连续id)
    result_indices = np.vstack(result_indices)
    result_rating_indices = np.vstack(result_rating_indices)
    return edges, result_indices, len(nodes), mapping, result_rating_indices


def parse_minibatch_LastFM(adjlists_ua, edge_metapath_indices_list_ua, rating_metapath_indices_list, user_artist_batch,
                           device, samples=None, use_masks=None, offset=None):
    g_lists = [[], []]  # item图和user图
    result_indices_lists = [[], []]  #
    result_rating_indices_lists = [[], []]  #
    idx_batch_mapped_lists = [[], []]
    for mode, (adjlists, edge_metapath_indices_list, rating_metapath_indices_) in enumerate(
            zip(adjlists_ua, edge_metapath_indices_list_ua, rating_metapath_indices_list)):
        # mode = 1 - mode  # 因为df user item顺序和id编号顺序不一致
        for adjlist, indices, use_mask, rating_metapath_indices in zip(adjlists, edge_metapath_indices_list,
                                                                       use_masks[mode], rating_metapath_indices_):
            if use_mask:
                edges, result_indices, num_nodes, mapping, result_rating_indices = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch],
                    [rating_metapath_indices[row[mode]] for row in user_artist_batch], samples,
                    exclude=user_artist_batch, offset=offset, mode=mode)
            else:
                edges, result_indices, num_nodes, mapping, result_rating_indices = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch],
                    [indices[row[mode]] for row in user_artist_batch],
                    [rating_metapath_indices[row[mode]] for row in user_artist_batch], samples, offset=offset,
                    mode=mode)

            g = dgl.DGLGraph(multigraph=True).to(device)
            g.add_nodes(num_nodes)
            if len(edges) > 0:  # 记得修改
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])  # 按照edge从小到大给id排序，输出id序列
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))  # ?
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
                result_rating_indices = torch.LongTensor(result_rating_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
                result_rating_indices = torch.LongTensor(result_rating_indices).to(device)

            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            result_rating_indices_lists[mode].append(result_rating_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in user_artist_batch]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists, result_rating_indices_lists


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def parse_minibatch_Ciao(origin_g, batch_size, metapath_list, train_pos_idx_batch, dataset_array, num_paths_per_node,
                         device,
                         samples=None, use_masks=None, offset=None):
    """

    :param samples:
    :param offset:
    :param use_masks:
    :param metapath_list: [[items' metapaths], [users' metapaths]]
    :param train_pos_idx_batch: 一batch的dataset_array行id
    :param dataset_array: np arrry格式的数据集，如train_ratings
    :param num_paths_per_node: 每个起点采样多少条特定路径
    :return:
    """
    # 生成计算图(同构图)，路径，映射
    g_list = [[], []]
    result_indices_lists = [[], []]
    rating_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    for mode, metapath_types in enumerate(metapath_list):  # item 0, user 1
        node_batch = torch.tensor(dataset_array[train_pos_idx_batch][:, mode].tolist())
        for metapath_type in metapath_types:
            edges, nodes, mapping, path_list, rating_list = parse_adjlist_Ciao(origin_g, batch_size,
                                                                               metapath_type, node_batch,
                                                                               num_paths_per_node, offset=offset,
                                                                               mode=mode)
            # 生成同构图  map ID映射  不需要加节点属性（节点emb之后用tensor取矩阵里的），不需要加评分属性，用于计算的同构图本就没有评分
            # if len(edges) == 0:
            #    continue
            g = dgl.graph(edges).to(device)  # ? 这样建图对吗
            g.add_nodes(len(nodes))
            path_list = torch.cat(path_list).to(device)
            rating_list = torch.cat(rating_list).to(device)
            """
            if len(edges) > 0:
                # sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])  # 按照edge从小到大给id排序，输出id序列
                # g.add_edges(*list(zip(*[(edges[j][0], edges[j][1]) for j in sorted_index])))  # ?
                result_indices = torch.LongTensor(path_list).to(device)  # ?
            else:
                result_indices = torch.LongTensor(path_list).to(device)
            """
            g_list[mode].append(g)
            result_indices_lists[mode].append(path_list)
            rating_indices_lists[mode].append(rating_list)
            if mode:
                idx_batch_mapped_lists[mode].append(
                    np.array([mapping[row[mode] + offset] for row in dataset_array[train_pos_idx_batch]]))
            else:
                idx_batch_mapped_lists[mode].append(
                    np.array([mapping[row[mode]] for row in dataset_array[train_pos_idx_batch]]))  # 报错 可能是偏置的问题
    return g_list, result_indices_lists, rating_indices_lists, idx_batch_mapped_lists


def parse_adjlist_Ciao(origin_g, batch_size, metapath_type, node_batch, num_paths_per_node, samples=None, exclude=None,
                       offset=None,
                       mode=None):
    """
    采样每种元路径 如（0，1，0）的路径
    :param batch_size:
    :param origin_g: 原始图 带有
    :param node_batch: 一batch的输入节点（随机游走的起点），也是这个batch训练更新emb的节点
    :param metapath_type: 如['rated-by', 'trust', 'rated']
    :param num_paths_per_node: 每个游走起点（1 batch的输入节点）采样路径数
    :param samples:
    :param exclude:
    :param offset: num_item
    :param mode:
    :return:
    """
    path_list = []
    rating_list = []
    start = []
    end = []
    nodes = []
    # num_paths_per_node = 12
    node_batch_ = [node for node in node_batch for _ in range(num_paths_per_node)]  # 每个节点每种元路径采样6条路径
    path00 = dgl.sampling.random_walk(
        origin_g, node_batch_, metapath=metapath_type)  # 一个起点一条指定路径
    # print(path00)  # 1 0 1 1 0 user 1 没有加偏置的id  从0开始 之后调用emb时user需要加偏置
    path = path00[0].view(len(node_batch), num_paths_per_node, path00[1].shape[0])  # 5是路径长度 最后一个batch不足batch_size会报错
    # print(path)

    offset_column_idx = torch.nonzero(path00[1]).squeeze()  # 需要加偏置的列id
    for paths_of_a_node in path:  # user加偏置  做map  生成同构图
        mask = (paths_of_a_node == -1).any(dim=1)
        idx = torch.nonzero(~mask).squeeze()
        nodes.append(paths_of_a_node[:, (0, -1)].view(-1))
        path_of_a_node_without_padding = torch.index_select(paths_of_a_node, dim=0, index=idx)  # 选择不含-1的行
        if path_of_a_node_without_padding.shape[0] == 0:
            continue
        # 评分 用path00[1]和origin_g 取评分  之前是用edges 直接在子图里取1batch的边属性张量 这里还是只能用评分字典取
        # 你可以用dgl.edge_ids函数来获取路径对应的边的ID
        ratings_of_a_node = torch.zeros(
            (path_of_a_node_without_padding.shape[0], path_of_a_node_without_padding.shape[1] - 1))
        type_list = path00[1].tolist()
        for i, node_type in enumerate(type_list):
            if i == len(type_list) - 1:
                continue
            if (node_type, type_list[i + 1]) == (1, 1):
                continue
            elif (node_type, type_list[i + 1]) == (1, 0):
                edge_ids = origin_g.edge_ids(path_of_a_node_without_padding[:, i],
                                             path_of_a_node_without_padding[:, i + 1], etype=('uid', 'rated', 'iid'))
                ratings_of_a_node[:, i] = origin_g.edata['rating'][('uid', 'rated', 'iid')][edge_ids]
                # ratings_of_a_node[:, i] = edge_ids

            else:
                edge_ids = origin_g.edge_ids(path_of_a_node_without_padding[:, i + 1],
                                             path_of_a_node_without_padding[:, i], etype=('uid', 'rated', 'iid'))
                ratings_of_a_node[:, i] = origin_g.edata['rating'][('uid', 'rated', 'iid')][edge_ids]
                # ratings_of_a_node[:, i] = edge_ids
        # 然后你可以用g.edata['rating'][edge_ids]来获取边属性
        path_of_a_node_without_padding[:, offset_column_idx] = torch.add(
            path_of_a_node_without_padding[:, offset_column_idx], offset)  # user加偏置
        start.append(path_of_a_node_without_padding[:, 0])
        end.append(path_of_a_node_without_padding[:, -1])
        path_list.append(path_of_a_node_without_padding)
        rating_list.append(torch.LongTensor(ratings_of_a_node.numpy()))  #
    nodes = torch.unique(torch.cat(nodes))
    if mode:
        nodes = torch.add(nodes, offset)
    nodes = nodes.tolist()
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}  # 根据本次采样节点出现顺序重新编号
    if len(start) == 0:
        edges = []
        nodes = []
    else:
        start = torch.cat(start)
        end = torch.cat(end)
        edges = torch.cat((start.unsqueeze(1), end.unsqueeze(1)), dim=1).tolist()
        edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))  # map 对edges所有元素使用lambda匿名函数，tup输入

    return edges, nodes, mapping, path_list, rating_list


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        dis = (yhat - y)
        mae = torch.sum(torch.abs(dis), dim=0) / (y.shape[0])
        return torch.sum(torch.pow(dis, 2), dim=0) / (y.shape[0]), mae


def parse_adjlist_Ciao_homo(origin_g, pathlen, node_batch, num_paths_per_node, offset=None, mode=None,
                            restart_prob=0.2):
    """
    采样每种元路径 如（0，1，0）的路径
    :param origin_g: 原始图 带有
    :param node_batch: 一batch的输入节点（随机游走的起点），也是这个batch训练更新emb的节点
    :param metapath_type: 如['rated-by', 'trust', 'rated']
    :param num_paths_per_node: 每个游走起点（1 batch的输入节点）采样路径数
    :param offset: num_item
    :param mode:
    :return:
    """
    path_list = []
    time_list = []
    start = []
    end = []
    nodes = []
    node_batch_ = [node for node in node_batch for _ in range(num_paths_per_node)]  # 每个节点每种元路径采样6条路径
    path00 = dgl.sampling.random_walk(origin_g, node_batch_, length=pathlen, return_eids=True,
                                      restart_prob=restart_prob)  # 一个起点一条指定路径 没有合适节点可能会跳2步
    path = path00[0].view(len(node_batch), num_paths_per_node, path00[2].shape[0])  # 5是路径长度 最后一个batch不足batch_size会报错
    edge = path00[1].view(len(node_batch), num_paths_per_node, path00[2].shape[0] - 1)
    for paths_of_a_node, edge_of_a_node in zip(path, edge):  # user加偏置  做map  生成同构图
        mask = (edge_of_a_node == -1).any(dim=1)
        idx = torch.nonzero(~mask).squeeze()
        nodes.append(paths_of_a_node[:, (0, -1)].view(-1))
        path_of_a_node_without_padding = torch.index_select(paths_of_a_node, dim=0, index=idx)  # 选择不含-1的行
        path_of_a_node_without_padding = torch.flip(path_of_a_node_without_padding, [1])  # 逆序  最后一个节点才是中心节点
        edge_of_a_node = torch.index_select(edge_of_a_node, dim=0, index=idx)
        edge_of_a_node = torch.flip(edge_of_a_node, [1])
        if path_of_a_node_without_padding.shape[0] == 0:
            continue
        # 评分 用path00[1]和origin_g 取评分  之前是用edges 直接在子图里取1batch的边属性张量 这里还是只能用评分字典取
        times_of_a_node = torch.zeros(
            (path_of_a_node_without_padding.shape[0], path_of_a_node_without_padding.shape[1] - 1))
        for i in range(path00[2].shape[0] - 1):
            # edge_ids = origin_g.edge_ids(path_of_a_node_without_padding[:, i], path_of_a_node_without_padding[:, i+1])
            times_of_a_node[:, i] = origin_g.edata['time'][edge_of_a_node[:, i]]
        # 然后你可以用g.edata['rating'][edge_ids]来获取边属性
        start.append(path_of_a_node_without_padding[:, 0])
        end.append(path_of_a_node_without_padding[:, -1])
        path_list.append(path_of_a_node_without_padding)
        time_list.append(torch.LongTensor(times_of_a_node.numpy()))  #
    nodes = torch.unique(torch.cat(nodes))
    nodes = nodes.tolist()
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}  # 根据本次采样节点出现顺序重新编号
    if len(start) == 0:
        edges = []
        nodes = []
    else:
        start = torch.cat(start)
        end = torch.cat(end)
        edges = torch.cat((start.unsqueeze(1), end.unsqueeze(1)), dim=1).tolist()
        edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))  # map 对edges所有元素使用lambda匿名函数，tup输入

    return edges, nodes, mapping, path_list, time_list


def parse_minibatch_Ciao_homo(origin_g, pathlen_lists, train_pos_idx_batch, dataset_array, num_paths_per_node, device,
                              offset=None, restart_prob=0.2):
    """

    :param restart_prob:
    :param samples:
    :param offset:
    :param use_masks:
    :param pathlen_lists: [[items' metapaths], [users' metapaths]]
    :param train_pos_idx_batch: 一batch的dataset_array行id
    :param dataset_array: np arrry格式的数据集，如train_ratings
    :param num_paths_per_node: 每个起点采样多少条特定路径
    :return:
    """
    # 生成计算图(同构图)，路径，映射
    g_list = [[], []]
    result_indices_lists = [[], []]
    time_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    for mode, pathlen_list in enumerate(pathlen_lists):  # item 0, user 1
        node_batch = torch.tensor(dataset_array[train_pos_idx_batch][:, mode].tolist())
        for pathlen in pathlen_list:
            edges, nodes, mapping, path_list, time_list = parse_adjlist_Ciao_homo(origin_g,
                                                                                  pathlen, node_batch,
                                                                                  num_paths_per_node, offset=offset,
                                                                                  mode=mode, restart_prob=restart_prob)
            # 生成同构图  map ID映射  不需要加节点属性（节点emb之后用tensor取矩阵里的），不需要加评分属性，用于计算的同构图本就没有评分
            # if len(edges) == 0:
            #    continue
            g = dgl.graph(edges).to(device)  # ? 这样建图对吗
            g.add_nodes(len(nodes))
            path_list = torch.cat(path_list).to(device)
            time_list = torch.cat(time_list).to(device)
            g_list[mode].append(g)
            result_indices_lists[mode].append(path_list)
            time_indices_lists[mode].append(time_list)
            idx_batch_mapped_lists[mode].append(
                np.array([mapping[row[mode]] for row in dataset_array[train_pos_idx_batch]]))  # 报错 可能是偏置的问题
    return g_list, result_indices_lists, time_indices_lists, idx_batch_mapped_lists
