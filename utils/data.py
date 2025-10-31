import networkx as nx
import numpy as np
import pandas as pd
import scipy
import pickle
import h5py

# 试验导入
from utils.tools import index_generator, parse_adjlist_LastFM, parse_minibatch


def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G01 = nx.read_adjlist(prefix + '/0/0-2-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    G11 = nx.read_adjlist(prefix + '/1/1-0-2-0-1.adjlist', create_using=nx.MultiDiGraph)
    G20 = nx.read_adjlist(prefix + '/2/2-0-2.adjlist', create_using=nx.MultiDiGraph)
    G21 = nx.read_adjlist(prefix + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx01 = np.load(prefix + '/0/0-2-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    idx11 = np.load(prefix + '/1/1-0-2-0-1_idx.npy')
    idx20 = np.load(prefix + '/2/2-0-2_idx.npy')
    idx21 = np.load(prefix + '/2/2-0-1-0-2_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [[G00, G01], [G10, G11], [G20, G21]], \
           [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
           [features_0, features_1, features_2], \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02[3:]
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-3-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3], \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx


def load_LastFM_data(prefix='data/preprocessed/LastFM_processed'):
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx12 = pickle.load(in_file)
    in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_user_artist.npz')
    train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_user_artist.npz')

    return [[adjlist00, adjlist01, adjlist02], [adjlist10, adjlist11, adjlist12]], \
           [[idx00, idx01, idx02], [idx10, idx11, idx12]], \
           adjM, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist


def load_Ciao_data(prefix='data/preprocessed/Ciao_processed'):  # ../data/preprocessed/Ciao_processed
    """
    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-1-0-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '/0/0-1-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + '/1/1-0-1-0-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-0-1-1-0-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()
    in_file = open(prefix + '/1/1-1.adjlist', 'r')
    adjlist13 = [line.strip() for line in in_file]
    adjlist13 = adjlist13
    in_file.close()

    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-0-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1-0-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '/1/1-0-1-1-0-1_idx.pickle', 'rb')
    # idx12 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '/1/1-1_idx.pickle', 'rb')
    idx13 = pickle.load(in_file)
    in_file.close()
    """
    """
    in_file = open(prefix + '/0/0-1-0_rating.pickle', 'rb')  # 取rating
    rating00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-0-1-0_rating.pickle', 'rb')
    rating01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0/0-1-1-0_rating.pickle', 'rb')
    rating02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_rating.pickle', 'rb')
    rating10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/1/1-0-1-0-1_rating.pickle', 'rb')
    rating11 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '/1/1-0-1-1-0-1_rating.pickle', 'rb')
    # rating12 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '/1/1-1_rating.pickle', 'rb')
    rating13 = pickle.load(in_file)
    in_file.close()
    """
    # adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_ratings = np.load(prefix + '/train_val_test_ratings.npz')
    # train_val_test_ratings_dict = np.load(prefix + '/train_val_test_ratings_dict.npy', allow_pickle=True)
    # rating_metapath_indices_list = np.load(prefix + '/rating_metapath_indices_list_no_social.npy', allow_pickle=True)

    return type_mask, train_val_test_ratings  # , train_val_test_ratings_dict

    # [[adjlist00, adjlist01, adjlist02], [adjlist10, adjlist11, adjlist13]], \
           # [[idx00, idx01, idx02], [idx10, idx11, idx13]], \
           # [[rating00, rating01, rating02], [rating10, rating11, rating13]]


# load skipgram-format embeddings, treat missing node embeddings as zero vectors
def load_skipgram_embedding(path, num_embeddings):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings = np.zeros((num_embeddings, dim))
        for line in infile.readlines():
            count += 1
            line = line.strip().split(' ')
            embeddings[int(line[0])] = np.array(list(map(float, line[1:])))
    print('{} out of {} nodes have non-zero embeddings'.format(count, num_embeddings))
    return embeddings


# load metapath2vec embeddings
def load_metapath2vec_embedding(path, type_list, num_embeddings_list, offset_list):
    count = 0
    with open(path, 'r') as infile:
        _, dim = list(map(int, infile.readline().strip().split(' ')))
        embeddings_dict = {type: np.zeros((num_embeddings, dim)) for type, num_embeddings in
                           zip(type_list, num_embeddings_list)}
        offset_dict = {type: offset for type, offset in zip(type_list, offset_list)}
        for line in infile.readlines():
            line = line.strip().split(' ')
            # drop </s> token
            if line[0] == '</s>':
                continue
            count += 1
            embeddings_dict[line[0][0]][int(line[0][1:]) - offset_dict[line[0][0]]] = np.array(
                list(map(float, line[1:])))
    print('{} node embeddings loaded'.format(count))
    return embeddings_dict


def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'data/wordvec/GloVe/glove.6B.50d.txt',
        100: 'data/wordvec/GloVe/glove.6B.100d.txt',
        200: 'data/wordvec/GloVe/glove.6B.200d.txt',
        300: 'data/wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs


def creat_rating_dict(df, item_id_mapping, user_id_mapping):
    rating_dict = {}
    for _, row in df.iterrows():
        rating_dict[(user_id_mapping[row['user']], item_id_mapping[row['item']])] = row['rating']
        rating_dict[(item_id_mapping[row['item']], user_id_mapping[row['user']])] = row['rating']
    return rating_dict


def creat_path_rating(rating_dict, edge_metapath_indices_list, num_item):
    # 遍历评分字典构建和路径对应的评分array
    save_prefix = '../data/preprocessed/Ciao_processed/'
    rating_metapath_indices_list = []
    for node_type, edge_metapath_indices in enumerate(edge_metapath_indices_list):
        rating_metapath_indices = []
        for _, edge_metapath_indice in np.ndenumerate(edge_metapath_indices):
            rating_metapath_indice = []
            for node, paths in edge_metapath_indice.items():
                rating_paths = []
                for path in paths:
                    tmp = []
                    for i, path_node in np.ndenumerate(path):
                        i = i[0]
                        if not i == len(path) - 1:
                            if not (path_node >= num_item and path[i+1] >= num_item):  # user user 社交关系无评分
                                tmp.append(rating_dict[(path_node, path[i+1])])
                            else:
                                tmp.append(0)
                    rating_paths.append(tmp)
                rating_metapath_indice.append(rating_paths)  # 一种路径
            with open(save_prefix + '{}/'.format(node_type) + '-'.join(map(str, edge_metapath_indice[0][0])) + '_rating.pickle',
                      'wb') as out_file:
                pickle.dump(np.array(rating_metapath_indice), out_file)

            rating_metapath_indices.append(rating_metapath_indice)  # user 的四种路径
        rating_metapath_indices_list.append(rating_metapath_indices)  # user item
    # np.save(save_prefix + 'rating_metapath_indices_list_no_social.npy',  # 将一整个大array存在一个npy 耗费3.55G 而且用了一下午 读取也很慢
    #           np.array(rating_metapath_indices_list))
    return rating_metapath_indices_list


def save_hdf5(data, prefix='data/preprocessed/Ciao_processed/'):
    f = h5py.File(prefix+'rating_idx_lists', 'w')
    f.create_dataset(f'{data}', data=data)
    f.close()


def read_hdf5(fileName, prefix='data/preprocessed/Ciao_processed/'):
    f = h5py.File(prefix+'rating_idx_lists', 'r')
    a = f[f'{fileName}'][:]
    f.close()
    return a


if __name__ == '__main__':
    adjlists, edge_metapath_indices_list, _, type_mask, train_val_test_ratings, ratings_dict, rating_metapath_indices_list = load_Ciao_data()

    # print(pd.DataFrame(adjlists[0][0]))
    # print(len(adjlists))
    # print(edge_metapath_indices_list_ua)
    train_ratings = train_val_test_ratings['train_ratings']
    # train_ratings_dict = ratings_dict['train_ratings']
    # print(train_ratings_dict)
    # print(len(edge_metapath_indices_list))
    # print(edge_metapath_indices_list[0][0][0])  # 终点为item 0的所有 user-item-user路径  [    0 16861     0] item和user id都从0开始
    # edge_metapath_indices_list[0][0][0] += 1

    # print(type(edge_metapath_indices_list[0][0][0][0]))  # 类型为np.ndarray
    # print(ratings_dict.item())  #
    rating_metapath_indices_list = creat_path_rating(ratings_dict.item(), edge_metapath_indices_list, 16861)
    print(rating_metapath_indices_list)


                # for j in range(path.size()-1):


            # edge_metapath_indice += 1  # item和user id都从0开始  看取emb是否从0开始
            # print(len(edge_metapath_indice))  # 之后 还需要采样  采样后再从字典取数 训练时采样很耗时  如果事先搞一个一样的list，采样时根据采样的index取值
            # 先看采样过程 是否方便取index -> 搞一个一样的评分list
            # for path in edge_metapath_indice:
                # print(path)
                # for i, node in enumerate(path.tolist()):
                    # print(i)
                    # print(node)




    ("\n"
     "    # 之前测试用的\n"
     "    train_user_item = pd.DataFrame(train_ratings)\n"
     "    train_user_item.columns = ['user', 'item', 'genreID', 'rating', 'helpfulness', 'date']\n"
     "    train_user_item = train_user_item[['item', 'user']].to_numpy()  # 数据第0个是user 但我的定义里0代表item 做一下替换\n"
     "    # print(train_user_item)\n"
     "    train_pos_idx_generator = index_generator(batch_size=1024, num_data=len(train_ratings))\n"
     "    num_user = 2378\n"
     "    num_item = 16861\n"
     "    offset = num_item\n"
     "    for iteration in range(train_pos_idx_generator.num_iterations()):\n"
     "        train_pos_idx_batch = train_pos_idx_generator.next()\n"
     "        train_pos_idx_batch.sort()  # 训练集的id\n"
     "        idx_batch = train_user_item[train_pos_idx_batch].tolist()\n"
     "        # print(train_pos_idx_batch)\n"
     "        print(idx_batch)  # [[818, 11178, 5, 5, 1, 1140678000], [403, 3666, 2, 5, 1, 1301814000],\n"
     "\n"
     "        for mode, (adjlist, indices) in enumerate(zip(adjlists, edge_metapath_indices_list)):\n"
     "            for adjl, indice in zip(adjlist, indices):\n"
     "                # print(adjlist[0][0])  # [['id id id '], ]  即[adjlist00, adjlist01, adjlist02]\n"
     "                # print(len(adjlist))  # len(adjlist) = num_item  每个item的(0, 1, 0)路径邻居(路径终点)   len(adjlist)=3\n"
     "                # [[adjlist00, adjlist01, adjlist02],[adjlist10, adjlist11, adjlist12, adjlist13]]\n"
     "                # print(indices[0][0])  # 路径\n"
     "                # print(idx_batch[0][0])\n"
     "                # print(mode)  # 0 item 1 user\n"
     "                print(adjl)\n"
     "                print(indice)\n"
     "                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(\n"
     "                    [adjl[row[mode]] for row in idx_batch], [indice[row[mode]] for row in idx_batch], 100,\n"
     "                    offset=offset, mode=mode)\n"
     "                # 采样邻居  从路径邻居里面选择邻居  生成edges 新的计算图\n"
     "                # print(edges)\n"
     "                # print(result_indices)  # 采样后的路径 每种元路径\n"
     "                # print(num_nodes)  # 2962\n"
     "                # print(mapping)  # 3: 0, 6: 1, 8: 2, 12: 3 # # 根据本次采样节点出现顺序重新编号\n"
     "    # train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(\n"
     "    #    adjlists_ua, edge_metapath_indices_list_ua, train_pos_user_artist_batch, device, neighbor_samples, use_masks,\n"
     "    #    num_item)\n"
     "    \n"
     "    ")