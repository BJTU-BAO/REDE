import torch, os
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets as geo_data
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse import diags
device = torch.device('cuda')
DATA_ROOT = 'data'
if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)


def load_data(data_name='cora', normalize_feature=True, missing_rate=0, citation_random=False, train_size=20,
              cuda=False):
    # can use other dataset, some doesn't have mask
    if data_name in ['cora', 'citeseer', 'pubmed']:
        data = geo_data.Planetoid(os.path.join(DATA_ROOT, data_name), data_name).data
        if citation_random:
            random_coauthor_amazon_splits(data, max(data.y), None, train_size)
    elif data_name in ['Photo', 'Computers']:
        data = geo_data.Amazon(os.path.join(DATA_ROOT, data_name), data_name, T.NormalizeFeatures()).data
        random_coauthor_amazon_splits(data, max(data.y) + 1, None)
    elif data_name in ['cora_ml', 'dblp']:
        data = geo_data.CitationFull(os.path.join(DATA_ROOT, data_name), data_name, T.NormalizeFeatures()).data
        random_coauthor_amazon_splits(data, max(data.y) + 1, None, train_size)
    else:
        data = geo_data.WikiCS(os.path.join(DATA_ROOT, data_name), data_name, T.NormalizeFeatures()).data
        data.train_mask = data.train_mask.type(torch.bool)[:, 0]
        data.val_mask = data.val_mask.type(torch.bool)[:, 0]
    print(max(data.y))
    # original split
    data.train_mask = data.train_mask.type(torch.bool)
    data.val_mask = data.val_mask.type(torch.bool)
    data.test_mask = data.test_mask.type(torch.bool)
    # data.test_mask = data.test_mask.type(torch.bool)

    # expand test_mask to all rest nodes
    # data.test_mask = ~(data.train_mask + data.val_mask)

    # get adjacency matrix
    n = len(data.x)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    data.degree_martix = to_torch_sparse(sp.diags(np.array(adj.sum(axis=1)).flatten()))

    data.noeye_adj = to_torch_sparse(normalize_adj(adj - sp.eye(adj.shape[0], adj.shape[1]))).to(device)
    # data.adj_origin = to_torch_sparse(adj)  # Adjacency matrix
    data.adj_origin = to_torch_sparse(adj)  # Adjacency matrix

    # middle-normalize,
    mid_adj = middle_normalize_adj(adj)
    data.mid_adj = to_torch_sparse(mid_adj)
    # exit()

    adj = normalize_adj(adj)  # symmetric normalization works bad, but why? Test more.
    data.adj = to_torch_sparse(adj)

    # normalize feature
    if normalize_feature:
        data.x = row_l1_normalize(data.x)

    # generate missing feature setting
    indices_dir = os.path.join(DATA_ROOT, data_name, 'indices')
    if not os.path.isdir(indices_dir):
        os.mkdir(indices_dir)
    missing_indices_file = os.path.join(indices_dir, "indices_missing_rate={}.npy".format(missing_rate))
    if not os.path.exists(missing_indices_file):
        erasing_pool = torch.arange(n)[~data.train_mask]  # keep training set always full feature
        size = int(len(erasing_pool) * (missing_rate / 100))
        idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
        np.save(missing_indices_file, idx_erased)
    else:
        idx_erased = np.load(missing_indices_file)
    # erasing feature for random missing
    if missing_rate > 0:
        data.x[idx_erased] = 0

    if cuda:
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.adj = data.adj.to(device)
        data.mid_adj = data.mid_adj.to(device)
        data.adj_origin = data.adj_origin.to(device)
        data.degree_martix = data.degree_martix.to(device)

    return data


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def adjPrecess(adj, features,edge_diff_matrix, edge_diff_matrix_high):

    fea_copy = features.toarray()
    sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)
    # Convert the adjacency matrix to a COO format
    adj_coo = adj.tocoo()
    adj_ori = adj_coo.copy()
    original_adj = sp.coo_matrix((adj_coo.data.copy(), (adj_coo.row.copy(), adj_coo.col.copy())), shape=adj.shape)
    original_nonzero_count = adj_coo.data.nonzero()[0].size

    diff = edge_diff_matrix[adj_coo.row, adj_coo.col]
    sorted_diff = np.sort(diff)[::-1]
    percentile_index = int(0.2 * len(sorted_diff))
    top_20_percent_values = sorted_diff[percentile_index]

    #Update the values based on the similarity threshold
    update_mask = (sim_matrix < 0.1) & (edge_diff_matrix >= top_20_percent_values)
    adj_coo.data[update_mask[adj_coo.row, adj_coo.col]] = 0

    update_mask_high = (sim_matrix >= 0.1) | (edge_diff_matrix < top_20_percent_values)
    adj_high = adj_ori.copy()
    adj_high.data[update_mask_high[adj_high.row, adj_high.col]] = 0
    updated_adj_high = sp.coo_matrix((adj_high.data, (adj_high.row, adj_high.col)), shape=adj.shape)
    updated_adj_high = updated_adj_high + updated_adj_high.T  # Ensure it's a symmetric matrix

    updated_adj_high.data[updated_adj_high.data > 1] = 1
    edges_count = updated_adj_high.data.nonzero()[0].size // 2

    # Reconstruct the updated adjacency matrix
    updated_adj = sp.coo_matrix((adj_coo.data, (adj_coo.row, adj_coo.col)), shape=adj.shape)
    updated_adj = updated_adj + updated_adj.T  # Ensure it's a symmetric matrix
    updated_adj.data[updated_adj.data > 1] = 1

    edges_count = updated_adj.data.nonzero()[0].size // 2

    return updated_adj, updated_adj_high, sim_matrix


def process(adj, features, labels, idx_train, idx_val, idx_test, alpha, edge_diff_matrix, edge_diff_matrix_high):

    data = Data()
    update_adj, updated_adj_high, sim = adjPrecess(adj, features, edge_diff_matrix, edge_diff_matrix_high)
    sp.save_npz("REDEAdj.npz", update_adj)
    up_adj = update_adj.copy()

    mid_adj, _ = middle_normalize_adj(update_adj, sim, alpha)
    _, low_adj = middle_normalize_adj(update_adj, sim, alpha=0.5)
    high_adj = middle_normalize_adj_high(updated_adj_high, sim, alpha=0.5)
    ##
    norm_adj = normalize_adj(adj)
    mid_adj = to_torch_sparse(mid_adj).to(device)
    ##
    low_adj = to_torch_sparse(low_adj).to(device)
    high_adj = to_torch_sparse(high_adj).to(device)
    ##
    norm_adj = to_torch_sparse(norm_adj).to(device)
    features = torch.tensor(features.A).float().to(device)
    # if fea == 0:
    features = row_l1_normalize(features)
    labels = torch.tensor(labels).long().to(device)

    data.mid_adj = mid_adj
    data.low_adj = low_adj
    data.high_adj = high_adj
    data.adj = norm_adj
    data.adj_origin = to_torch_sparse(adj).to(device)
    data.update_adj_origin = update_adj
    data.update_adj_origin_high = updated_adj_high
    data.x = features
    data.y = labels

    data.train_mask = idx_train
    data.val_mask = idx_val
    data.test_mask = idx_test

    return data


def compute_weight_matrix(adj_matrix, sim, threshold):

    degrees = np.array(adj_matrix.sum(axis=1)).flatten()

    R = threshold
    B_data = np.minimum(R, degrees) / R * sim

    adj_matrix = adj_matrix + diags([np.ones(adj_matrix.shape[0])], [0], format='csr')
    rows, cols = adj_matrix.nonzero()
    B = csr_matrix((B_data[rows, cols], (rows, cols)), shape=sim.shape)
    row_sum = B.sum(axis=1).A.flatten()
    row_sum[row_sum == 0] = 1
    normalized_weights = csr_matrix(B / row_sum[:, np.newaxis])

    return normalized_weights


def  middle_normalize_adj(adj, sim, alpha):
    """Middle normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot
    rowsum = np.array(adj.sum(1))
    if 0 in rowsum:
        rowsum[rowsum == 0] = 1e-12
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    DAD = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    low_a = compute_weight_matrix(adj, sim, threshold=10)
    mid_a = (alpha * sp.eye(adj.shape[0], adj.shape[1]) - DAD).dot(sp.eye(adj.shape[0], adj.shape[1]) + DAD)

    return mid_a, low_a

def  middle_normalize_adj_high(adj, sim, alpha):
    """Middle normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot
    rowsum = np.array(adj.sum(1))
    if 0 in rowsum:
        rowsum[rowsum == 0] = 1e-12
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    DAD = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    high_a = - DAD

    return high_a

def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx


def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X / norm


def dense_tensor2sparse_tensor(adj):
    indices = torch.nonzero(adj != 0)
    indices = indices.t()
    re_adj = torch.reshape(adj, (-1, 1))
    nonZeroRows = torch.abs(re_adj).sum(dim=1) != 0
    re_adj = re_adj[nonZeroRows]
    value = re_adj.t().squeeze()
    shape = torch.Size(adj.shape)
    new_adj = torch.sparse_coo_tensor(indices, value, shape)
    return new_adj


def random_coauthor_amazon_splits(data, num_classes, lcc_mask, train_size=20):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)



    train_index = torch.cat([i[:train_size] for i in indices], dim=0)
    val_index = torch.cat([i[train_size:train_size + 30] for i in indices], dim=0)

    rest_index = torch.cat([i[train_size + 30:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


from sklearn.metrics import roc_auc_score, f1_score

def compute_auc(output, labels):
    # Convert logits to probabilities using softmax
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()

    # Convert labels to numpy array
    labels = labels.detach().cpu().numpy()

    # Use roc_auc_score with One-vs-Rest (OvR) approach for multi-class AUC
    auc = roc_auc_score(labels, probs, multi_class='ovr')

    return auc


def compute_f1(output, labels):
    # Get the predicted class by taking the argmax of the output logits
    preds = output.max(1)[1].detach().cpu().numpy()

    # Convert labels to numpy array
    labels = labels.detach().cpu().numpy()

    # Calculate F1-score with macro averaging
    f1 = f1_score(labels, preds, average='macro')

    return f1

class Data():
    def __init__(self):
        self.A = None

if __name__ == "__main__":
    import sys

    print(sys.version)
    # test goes here
    data = load_data(cuda=True)
    print(data.train_mask[:150])