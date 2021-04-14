import os

import torch

from torch_geometric.utils import train_test_split_edges

import math
import torch
from torch_geometric.utils import to_undirected


def my_train_test_split_edges(data, val_ratio=0.15, test_ratio=0.05):

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_tr = int(math.floor((1 - val_ratio - test_ratio) * row.size(0)))
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_pos_edge_index = to_undirected(data.val_pos_edge_index)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_pos_edge_index = to_undirected(data.test_pos_edge_index)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_tr + n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_tr], neg_col[:n_tr]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)
    data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)

    row, col = neg_row[n_tr:n_tr + n_v], neg_col[n_tr:n_tr + n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)
    data.val_neg_edge_index = to_undirected(data.val_neg_edge_index)

    row, col = neg_row[n_tr + n_v:n_tr + n_v + n_t], neg_col[n_tr + n_v:n_tr + n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
    data.test_neg_edge_index = to_undirected(data.test_neg_edge_index)

    return data


def link_pred_transform(data):
    num_labels = len(torch.unique(data.y))
    data.x = torch.zeros((data.x.shape[0], num_labels))
    for i, node_lab in enumerate(data.y):
        data.x[i, node_lab] = 1

    # the function adds attributes train_pos_edge_index, train_edge_index, val_pos_edge_index, 
    # val_neg_edge_index, test_pos_edge_index, and test_neg_edge_index to data.
    data = my_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)

    # compute number of edges
    num_pos_train_edges = data.train_pos_edge_index.shape[1]
    num_neg_train_edges = data.train_neg_edge_index.shape[1]
    num_pos_val_edges = data.val_pos_edge_index.shape[1]
    num_neg_val_edges = data.val_neg_edge_index.shape[1]
    num_pos_test_edges = data.test_pos_edge_index.shape[1]
    num_neg_test_edges = data.test_neg_edge_index.shape[1]

    # create attributes
    data.train_pos_edge_attr = torch.ones(num_pos_train_edges, 1)
    data.train_neg_edge_attr = torch.zeros(num_neg_train_edges, 1)
    data.val_pos_edge_attr = torch.ones(num_pos_val_edges, 1)
    data.val_neg_edge_attr = torch.zeros(num_neg_val_edges, 1)
    data.test_pos_edge_attr = torch.ones(num_pos_test_edges, 1)
    data.test_neg_edge_attr = torch.zeros(num_neg_test_edges, 1)
    
    return data
    

def shuffle_train_val(data):

    train_pos = data.train_pos_edge_index
    train_neg = data.train_neg_edge_index
    val_pos = data.val_pos_edge_index
    val_neg = data.val_neg_edge_index

    num_pos_train_edges = train_pos.shape[1] // 2
    num_neg_train_edges = train_neg.shape[1] // 2

    all_pos = torch.cat((train_pos, val_pos), dim=1)
    all_neg = torch.cat((train_neg, val_neg), dim=1)

    # Positive edges
    row, col = all_pos

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:num_pos_train_edges], col[:num_pos_train_edges]
    train_pos = torch.stack([r, c], dim=0)
    train_pos = to_undirected(train_pos)

    r, c = row[num_pos_train_edges:], col[num_pos_train_edges:]
    val_pos = torch.stack([r, c], dim=0)
    val_pos = to_undirected(val_pos)

    # Negative edges
    row, col = all_neg

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:num_neg_train_edges], col[:num_neg_train_edges]
    train_neg = torch.stack([r, c], dim=0)
    train_neg = to_undirected(train_neg)

    r, c = row[num_neg_train_edges:], col[num_neg_train_edges:]
    val_neg = torch.stack([r, c], dim=0)
    val_neg = to_undirected(val_neg)

    data.train_pos_edge_index = train_pos
    data.train_neg_edge_index = train_neg
    data.val_pos_edge_index = val_pos
    data.val_neg_edge_index = val_neg

    return data