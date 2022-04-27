import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, device,
                 activation, dropout):
        super(GCNNet, self).__init__()
        conv = GCNConv

        self.conv1 = conv(in_channels, hidden_dim)
        self.conv2 = conv(hidden_dim, out_channels)

        self.activation = F.relu if activation == 'relu' else F.elu
        self.dropout = dropout
        self.device = device

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.double()

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class MLPBaseline(torch.nn.Module):
    def __init__(self, in_dim, H, out_dim):
        super(MLPBaseline, self).__init__()
        self.input_linear = torch.nn.Linear(in_dim, H)
        self.mid_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, out_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28).double()
        x = F.relu(self.input_linear(x))
        x = self.dropout(x)
        x = F.relu(self.mid_linear(x))
        x = self.dropout(x)
        x = self.output_linear(x)

        return F.log_softmax(x, dim=1)
