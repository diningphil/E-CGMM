import torch

class ConstantNodeEdgeIfEmpty:
    r"""Adds a constant value to each node feature only if x is None.
    Args:
        value (int, optional): The value to add. (default: :obj:`1`)
    """

    def __init__(self, value=1):
        self.value = value

    def __call__(self, data):
        if data.x is None:
            c = torch.full((data.num_nodes, 1), self.value, dtype=torch.float)
            data.x = c

        if data.edge_attr is None:
            c = torch.full((data.edge_index.shape[1], 1), self.value, dtype=torch.float)
            data.edge_attr = c

        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)