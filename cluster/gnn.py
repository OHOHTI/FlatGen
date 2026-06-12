# GNN model definitions for the self-supervised sublattice encoder.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, LayerNorm, global_mean_pool


class EdgeGNNConv(MessagePassing):
    """
    Custom message-passing layer that concatenates neighbour node features
    with edge attributes before projecting.
    """

    def __init__(self, in_channels, out_channels, edge_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(in_channels + edge_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        edge_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.relu(x + edge_messages)

    def message(self, x_j, edge_attr):
        return self.lin_edge(torch.cat([x_j, edge_attr], dim=-1))


class GNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder with EdgeGNNConv, residual connections,
    mean pooling, and a projection head for graph-level embeddings.
    """

    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_dim=128,
        embedding_dim=64,
        num_layers=4,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = EdgeGNNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=num_edge_features,
                aggr='mean',
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_dim))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )

        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            x_residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x) + x_residual
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = global_mean_pool(x, batch)
        return self.fc(x)


# Backward-compatible alias so existing checkpoints / imports still work.
GNNWithoutHyperedges = GNNEncoder
