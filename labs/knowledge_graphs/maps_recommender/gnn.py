import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GNNModel(torch.nn.Module):
    """Graph Neural Network model for node classification"""

    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.1):
        super(GNNModel, self).__init__()
        self.conv1 = gnn.GCNConv(num_features, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third Graph Convolution Layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Linear layer for classification
        x = self.lin(x)

        return x
