import os
import sys
import json
import shutil
import torch
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_geometric.datasets import TUDataset, Planetoid, KarateClub, QM7b
from torch_geometric.io import read_tu_data
# Suppress that annoying "Using backend" DGL message caused by OBG and DGL
stderr_tmp = sys.stderr
null = open(os.devnull, 'w')
sys.stderr = null
from ogb.graphproppred import PygGraphPropPredDataset
from dgl.data.utils import load_graphs
sys.stderr = stderr_tmp


class ZipDataset(torch.utils.data.Dataset):
    """
    This Dataset takes n datasets and "zips" them. When asked for the i-th element, it returns the i-th element of
    all n datasets. The lenght of all datasets must be the same.
    """
    def __init__(self, *datasets):
        """
        Stores all datasets in an internal variable.
        :param datasets: An iterable with PyTorch Datasets
        """
        self.datasets = datasets

        assert len(set(len(d) for d in self.datasets)) == 1

    def __getitem__(self, index):
        """
        Returns the i-th element of all datasets
        :param index: the index of the data point to retrieve
        :return: a list containing one element for each dataset in the ZipDataset
        """
        return [d[index] for d in self.datasets]

    def __len__(self):
        return len(self.datasets[0])


class ConcatFromListDataset(InMemoryDataset):
    """Create a dataset from a `torch_geometric.Data` list.
    Args:
        data_list (list): List of graphs.
    """
    def __init__(self, data_list):
        super(ConcatFromListDataset, self).__init__("")
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class DatasetInterface:

    name = None

    @property
    def dim_node_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")

    @property
    def dim_edge_features(self):
        raise NotImplementedError("You should subclass DatasetInterface and implement this method")


class TUDatasetInterface(TUDataset, DatasetInterface):

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, use_node_attr=False, use_edge_attr=False, cleaned=False):
        print('PROVA')
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)

        if 'aspirin' in self.name:
            # For regression problems
            if len(self.data.y.shape) == 1:
                self.data.y = self.data.y.unsqueeze(1)
            self.data.y = self.data.y/100000.

        if 'alchemy_full' in self.name or 'QM9' in self.name:
            # For regression problems
            if len(self.data.y.shape) == 1:
                self.data.y = self.data.y.unsqueeze(1)

            # Normalize all target variables (just for stability purposes)
            mean = self.data.y.mean(0).unsqueeze(0)
            std = self.data.y.std(0).unsqueeze(0)
            self.data.y = (self.data.y - mean) / std

        if 'ZINC_full' in self.name:
            # For regression problems
            if len(self.data.y.shape) == 1:
                self.data.y = self.data.y.unsqueeze(1)

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        if 'alchemy_full' in self.name:
            return self.data.y.shape[1]
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()


class QM7bECGMMInterface(QM7b, DatasetInterface):

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat'

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        super().download()

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return 14

    def process(self):
        data = scipy.io.loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])    # n_graphs x max(num_nodes) x max(num_nodes)
        # coulomb_matrix[i] is the coulomb matrix in pos [:num_nodes[i], :num_nodes[i]] and 0 elsewhere
        target = torch.from_numpy(data['T']).to(torch.float)    # n_graphs x targets (14)
        # compute node_attr as list of n_graphs tensors num_nodes[i]
        node_attrs = []
        for i in range(target.shape[0]):
            node_attr = torch.diagonal(coulomb_matrix[i])
            node_attr = node_attr[torch.nonzero(node_attr, as_tuple=True)]
            node_attrs.append(node_attr)
        # get unique node attributes to map attrs to categorical
        unique_node_attrs = torch.unique(torch.cat(node_attrs, dim=0))
        dim_node_attr = len(unique_node_attrs)
        # set diagonals to 0
        coulomb_matrix = coulomb_matrix * (1 - torch.eye(coulomb_matrix.shape[1], coulomb_matrix.shape[2]).unsqueeze(0))
        # set the values bigger than treshold to 0
        treshold = 0.52197913
        coulomb_matrix = torch.where(coulomb_matrix < treshold, coulomb_matrix, torch.tensor(0.))

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero(
                as_tuple=False).t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            edge_attr = edge_attr.reshape(-1, 1)
            y = target[i].view(1, -1)
            num_nodes = node_attrs[i].shape[0]
            # remember node_attrs[i] tensor of shape (num_nodes)
            x = torch.zeros(num_nodes, dim_node_attr)
            for j, attr in enumerate(node_attrs[i]):
                x[j, torch.nonzero(attr==unique_node_attrs)] = 1
            # avoid graphs with no edges
            if edge_index.shape[1] == 0:
                continue
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            #data.num_nodes = edge_index.max().item() + 1
            data.num_nodes = num_nodes
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class QM7bCGMMInterface(QM7b, DatasetInterface):

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat'

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        super().download()

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return 14

    def process(self):
        data = scipy.io.loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])    # n_graphs x max(num_nodes) x max(num_nodes)
        # coulomb_matrix[i] is the coulomb matrix in pos [:num_nodes[i], :num_nodes[i]] and 0 elsewhere
        target = torch.from_numpy(data['T']).to(torch.float)    # n_graphs x targets (14)
        # compute node_attr as list of n_graphs tensors num_nodes[i]
        node_attrs = []
        for i in range(target.shape[0]):
            node_attr = torch.diagonal(coulomb_matrix[i])
            node_attr = node_attr[torch.nonzero(node_attr, as_tuple=True)]
            node_attrs.append(node_attr)
        # get unique node attributes to map attrs to categorical
        unique_node_attrs = torch.unique(torch.cat(node_attrs, dim=0))
        dim_node_attr = len(unique_node_attrs)
        # set diagonals to 0
        coulomb_matrix = coulomb_matrix * (1 - torch.eye(coulomb_matrix.shape[1], coulomb_matrix.shape[2]).unsqueeze(0))
        # set the values bigger than treshold to 0
        treshold = 0.52197913
        coulomb_matrix = torch.where(coulomb_matrix < treshold, coulomb_matrix, torch.tensor(0.))

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero(
                as_tuple=False).t().contiguous()
            #edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            #edge_attr = edge_attr.reshape(-1, 1)
            edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
            y = target[i].view(1, -1)
            num_nodes = node_attrs[i].shape[0]
            # remember node_attrs[i] tensor of shape (num_nodes)
            x = torch.zeros(num_nodes, dim_node_attr)
            for j, attr in enumerate(node_attrs[i]):
                x[j, torch.nonzero(attr==unique_node_attrs)] = 1
            # avoid graphs with no edges
            if edge_index.shape[1] == 0:
                continue
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            #data.num_nodes = edge_index.max().item() + 1
            data.num_nodes = num_nodes
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class QM7bCGMMDiscretizedInterface(QM7b, DatasetInterface):

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat'

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        super().download()

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return 14

    def process(self):
        data = scipy.io.loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])    # n_graphs x max(num_nodes) x max(num_nodes)
        # coulomb_matrix[i] is the coulomb matrix in pos [:num_nodes[i], :num_nodes[i]] and 0 elsewhere
        target = torch.from_numpy(data['T']).to(torch.float)    # n_graphs x targets (14)
        # compute node_attr as list of n_graphs tensors num_nodes[i]
        node_attrs = []
        for i in range(target.shape[0]):
            node_attr = torch.diagonal(coulomb_matrix[i])
            node_attr = node_attr[torch.nonzero(node_attr, as_tuple=True)]
            node_attrs.append(node_attr)
        # get unique node attributes to map attrs to categorical
        unique_node_attrs = torch.unique(torch.cat(node_attrs, dim=0))
        dim_node_attr = len(unique_node_attrs)
        # set diagonals to 0
        coulomb_matrix = coulomb_matrix * (1 - torch.eye(coulomb_matrix.shape[1], coulomb_matrix.shape[2]).unsqueeze(0))
        # set the values bigger than treshold to 0
        treshold = 0.52197913
        coulomb_matrix = torch.where(coulomb_matrix < treshold, coulomb_matrix, torch.tensor(0.))
        # coulomb matrix discretization
        n_bins = 10
        max_val = torch.max(coulomb_matrix)
        min_val = torch.min(coulomb_matrix[coulomb_matrix.nonzero(as_tuple=True)])
        bins_lenght = (max_val - min_val) / n_bins

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero(
                as_tuple=False).t().contiguous()
            edge_label = torch.floor_divide(coulomb_matrix[i, edge_index[0], edge_index[1]] - min_val, bins_lenght).long()
            edge_label[edge_label == n_bins] = n_bins - 1     # avoid max
            edge_attr=edge_label
            #
            y = target[i].view(1, -1)
            #
            num_nodes = node_attrs[i].shape[0]
            # remember node_attrs[i] tensor of shape (num_nodes)
            x = torch.zeros(num_nodes, dim_node_attr)
            for j, attr in enumerate(node_attrs[i]):
                x[j, torch.nonzero(attr==unique_node_attrs)] = 1
            if edge_index.shape[1] == 0:
                continue
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            #data.num_nodes = edge_index.max().item() + 1
            data.num_nodes = num_nodes
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PlanetoidDatasetInterface(Planetoid, DatasetInterface):

    # Do not implement a dummy init function that calls super().__init__, ow it breaks

    @property
    def dim_node_features(self):
        return self.num_features

    @property
    def dim_edge_features(self):
        return self.num_edge_features

    @property
    def dim_target(self):
        return self.num_classes

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def download(self):
        super().download()

    # Needs to be defined in each subclass of torch_geometric.data.Dataset
    def process(self):
        super().process()
