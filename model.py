import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from MultiGCN.layers import GraphConv, GraphAttConv
import numpy as np
import torch.nn.functional as F
import torch_sparse
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from MultiGCN.utils import middle_normalize_adj, to_torch_sparse, device, middle_normalize_adj_high

def adjPrecess(adj, adj_high,features,edge_diff_matrix, edge_diff_matrix_high):

    fea_copy = features.toarray()
    sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)
    # Convert the adjacency matrix to a COO format
    adj_coo = adj.tocoo()
    adj_coo_high = adj_high.tocoo()
    original_nonzero_count = adj_coo.data.nonzero()[0].size

    #update_mask = (sim_matrix < 0.1) & (edge_diff_matrix >= top_20_percent_values)
    update_mask = sim_matrix < 0.1
    adj_coo.data[update_mask[adj_coo.row, adj_coo.col]] = 0
    #
    # Reconstruct the updated adjacency matrix
    updated_adj = sp.coo_matrix((adj_coo.data, (adj_coo.row, adj_coo.col)), shape=adj.shape)
    #
    adj_coo_high.data[update_mask[adj_coo_high.row, adj_coo_high.col]] = 1
    updated_adj_high = sp.coo_matrix((adj_coo_high.data, (adj_coo_high.row, adj_coo_high.col)), shape=adj.shape)

    k = 7
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    num_nodes = updated_adj.shape[0]
    row_indices = np.repeat(np.arange(num_nodes), k)
    col_indices = top_k_indices.flatten()
    values = np.ones(num_nodes * k)
    T_k = csr_matrix((values, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    updated_adj = updated_adj + T_k

    updated_adj = updated_adj + updated_adj.T  # Ensure it's a symmetric matrix
    updated_adj.data[updated_adj.data > 1] = 1
    edges_count = updated_adj.data.nonzero()[0].size // 2

    return updated_adj, updated_adj_high, sim_matrix

class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, edge_diff_matrix, edge_diff_matrix_high, dropout, combine, nlayer=2, args=None):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i == 0 else nhid, nhid, bias=False)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer == 1 else nhid, nclass)

        self.addHid_layer = GraphConv(nfeat, nhid, bias=False)
        self.addOut_layer = GraphConv(nhid, nclass)

        self.addHid_layer_HIGH = GraphConv(nfeat, nhid, bias=False)
        self.addOut_layer_HIGH = GraphConv(nhid, nclass)

        self.combine = combine
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.nrom = nn.BatchNorm1d(nhid)

        self.gate2 = nn.Parameter(torch.tensor([0.5]))

        self.edge_diff_matrix = edge_diff_matrix
        self.edge_diff_matrix_high = edge_diff_matrix_high

        # self.relu = nn.GELU()


    def forward(self, x, data):
        adj = data.mid_adj
        adj2 = data.low_adj
        adj3 = data.high_adj
        up_adj = data.update_adj_origin
        up_adj_high = data.update_adj_origin_high
        x_ori = x
        # adj = data.adj# _origin
        # print(data.adj.to_dense())
        # exit()
        # new_adj = self._preprocess_adj(adj, normalize)
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)

            x = layer(x, adj)
            x = self.relu(x)

        x = self.dropout(x)
        x = self.out_layer(x, adj)

        x2 = self.dropout(x_ori)
        x2 = self.addHid_layer(x2, adj2)
        x2 = self.relu(x2)

        x_HIGH = self.dropout(x_ori)
        x_HIGH = self.addHid_layer_HIGH(x_HIGH, adj3)
        x_HIGH = self.relu(x_HIGH)

        x_hid = torch.cat((x2, x_HIGH), dim=1)

        # up_adj_l1, up_adj_l1_high, sim = adjPrecess(up_adj, up_adj_high, sp.csr_matrix(x_hid.cpu().detach().numpy()), self.edge_diff_matrix, self.edge_diff_matrix_high)
        # _, low_adj = middle_normalize_adj(up_adj_l1, sim, alpha=0.5)
        # low_adj = to_torch_sparse(low_adj).to(device)
        # high_adj = middle_normalize_adj_high(up_adj_l1_high, sim, alpha=0.5)
        # high_adj = to_torch_sparse(high_adj).to(device)

        # ===================================

        x2 = self.dropout(x2)
        x2 = self.addOut_layer(x2, adj2)

        x_HIGH = self.dropout(x_HIGH)
        x_HIGH = self.addOut_layer_HIGH(x_HIGH, adj3)

        x_final = self.gate2 * x + (1-self.gate2) * (x2 + x_HIGH)
        x_final = torch.log_softmax(x_final, dim=-1)
        return [x_final, None]


