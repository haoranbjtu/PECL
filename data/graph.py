import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat
    
    @staticmethod
    def normalize_laplacian_matrix_rate(adj_mat):  
        """ 
        对邻接矩阵进行行归一化。  
        参数:  
        - adj_mat: scipy.sparse.csr_matrix, 邻接矩阵。  
        返回:  
        - 归一化后的邻接矩阵。  
        note:
        基于对称归一化(也称为拉普拉斯归一化),如果你仅仅想要行归一化(即每行权重之和为1),
        应该只应用左侧的对角矩阵=(d_inv_sqrt.dot(adj_mat))，但这会改变矩阵的对称性。
        我使用了scipy.sparse库来处理稀疏矩阵,这在处理大型图时非常有效。
        row_sums[row_sums == 0] = 1 这行代码是为了防止除以0的错误。如果你的图中没有孤立的节点(即所有节点至少与一个其他节点相连),
        可以省略这行代码。然而，在实际应用中，图中通常会有孤立的节点，因此这样的处理是有必要的。
        """  
        # 计算每行的和  
        row_sums = np.array(adj_mat.sum(axis=1)).flatten()  
        # 避免除以0的情况  
        row_sums[row_sums == 0] = 1  
        # 创建一个对角矩阵，用于归一化  
        d_inv_sqrt = sp.diags(np.power(row_sums, -0.5).flatten())  
        # 进行行归一化  
        normalized_adj = d_inv_sqrt.dot(adj_mat).dot(d_inv_sqrt)  
        return normalized_adj  
    
    @staticmethod
    def normalize_laplacian_matrix(adj_mat):  
        """ 
        对邻接矩阵进行行归一化。  
        参数:  
        - adj_mat: scipy.sparse.csr_matrix, 邻接矩阵。  
        返回:  
        - 归一化后的邻接矩阵。  
        note:
        基于对称归一化(也称为拉普拉斯归一化),如果你仅仅想要行归一化(即每行权重之和为1),
        应该只应用左侧的对角矩阵=(d_inv_sqrt.dot(adj_mat))，但这会改变矩阵的对称性。
        我使用了scipy.sparse库来处理稀疏矩阵,这在处理大型图时非常有效。
        row_sums[row_sums == 0] = 1 这行代码是为了防止除以0的错误。如果你的图中没有孤立的节点(即所有节点至少与一个其他节点相连),
        可以省略这行代码。然而，在实际应用中，图中通常会有孤立的节点，因此这样的处理是有必要的。
        """  
        # 计算每行的和  
        row_sums = np.array(adj_mat.sum(axis=1)).flatten()  
        # 避免除以0的情况  
        row_sums[row_sums == 0] = 1  
        # 创建一个对角矩阵，用于归一化  
        d_inv_sqrt = sp.diags(np.power(row_sums, -0.5).flatten())  
        # 进行行归一化  
        normalized_adj = d_inv_sqrt.dot(adj_mat).dot(d_inv_sqrt)  
        return normalized_adj  

    def convert_to_laplacian_mat(self, adj_mat):
        pass
