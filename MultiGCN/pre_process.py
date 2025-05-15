from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import *
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np

def edge_diff(adj):

    adj_matrix = adj
    degree_vector = np.array(adj_matrix.sum(axis=1)).flatten()
    D_sqrt_inv = sp.diags(1.0 / np.sqrt(degree_vector), format='csr')
    normalized_matrix = D_sqrt_inv.dot(adj_matrix).dot(D_sqrt_inv)
    eigenvalues, U = np.linalg.eig(normalized_matrix.toarray())
    real_eigenvalues = eigenvalues.real
    sorted_indices = np.argsort(real_eigenvalues)[::-1]
    sorted_eigenvalues = real_eigenvalues[sorted_indices]
    U = U[:, sorted_indices]

    end = normalized_matrix.shape[0] // 2
    num_nodes = normalized_matrix.shape[0]
    edge_diff_matrix = np.zeros((num_nodes, num_nodes))
    edge_diff_matrix_high = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:
                edge_singular_value_diff = 0
                for y in range(int(num_nodes/20), int(num_nodes/3)):
                    singular_value_diff_y = np.abs(-2 * U[i, y] * U[j, y] + sorted_eigenvalues[y] * (U[i, y] ** 2 + U[j, y] ** 2))
                    edge_singular_value_diff += singular_value_diff_y
                edge_diff_matrix[i, j] = edge_singular_value_diff

                edge_singular_value_diff = 0
                for y in range(int(2 * num_nodes/3), int(num_nodes)):
                    singular_value_diff_y = np.abs(-2 * U[i, y] * U[j, y] + sorted_eigenvalues[y] * (U[i, y] ** 2 + U[j, y] ** 2))
                    edge_singular_value_diff += singular_value_diff_y
                edge_diff_matrix_high[i, j] = edge_singular_value_diff

    edge_diff_matrix += edge_diff_matrix.T - np.diag(edge_diff_matrix.diagonal())
    edge_diff_matrix_high += edge_diff_matrix_high.T - np.diag(edge_diff_matrix_high.diagonal())
    file_name = "Cora/edge_diff_matrix_10.npy"
    np.save(file_name, edge_diff_matrix)
    file_name = "Cora/edge_diff_matrix_high_10.npy"
    np.save(file_name, edge_diff_matrix_high)
    print("preprocess completed...")

    return edge_diff_matrix, edge_diff_matrix_high

def main():

    perturbed_data_file = "../Data/cora_meta_adj_0.05.npz"
    modified_adj = sp.load_npz(perturbed_data_file)

    edge_diff_matrix, edge_diff_matrix_high = edge_diff(modified_adj)