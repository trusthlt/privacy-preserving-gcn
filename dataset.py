import os
import pickle

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.utils import add_self_loops, degree


class GCNDataset(Dataset):
    def __init__(self, root, filename, split_type='train', transform=None, pre_transform=None):

        with open(filename, 'rb') as in_f:
            self.data = pickle.load(in_f)

        self.split_type = split_type
        super(GCNDataset, self).__init__(root)

    @property
    def raw_file_names(self):
        return ['file1']

    @property
    def processed_file_names(self):
        return ['data_{}_{}.pt'.format(self.split_type, i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        # Will not run if data_{}_{}.pt files already exist
        nfs, es, efs, ys = prepare_graph_elements(self.data)
        for data_idx in range(len(nfs)):
            data = Data(x=nfs[data_idx], y=ys[data_idx],
                        edge_index=es[data_idx], edge_attr=efs[data_idx])
            torch.save(data, os.path.join('processed', 'data_{}_{}.pt'.format(self.split_type, data_idx)))

    def get(self, idx):
        data = torch.load(os.path.join('processed', 'data_{}_{}.pt'.format(self.split_type, idx)))
        return data


def prepare_graph_elements(data):
    nfs = [torch.tensor(data[0][a], dtype=torch.double) for a in range(len(data[0]))]
    es = [torch.tensor(data[1][a], dtype=torch.long).T for a in range(len(data[1]))]
    efs = [torch.tensor(data[2][a], dtype=torch.double) for a in range(len(data[2]))]
    ys = [torch.tensor(data[3][a], dtype=torch.long) for a in range(len(data[3]))]

    return nfs, es, efs, ys
