import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GNNModel(torch.nn.Module):
    """Graph Neural Network model for node classification"""

    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.2):
        super(GNNModel, self).__init__()
        # Reduce dimensionality first
        self.dim_reduce = torch.nn.Linear(num_features, hidden_channels)
        
        # Two GCN layers with residual connections
        self.conv1 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Output layer
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # Reduce input dimensionality
        x = self.dim_reduce(x)
        x = F.gelu(x)
        identity = x
        
        # First Graph Convolution Layer with residual
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual connection
        
        # Second Graph Convolution Layer with residual
        identity = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual connection

        # Linear layer for classification
        x = self.lin(x)
        
        return x
