import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
from typing import Optional, Tuple

""" same RBFExpansion as in https://github.com/usnistgov/alignn/blob/main/alignn/models/utils.py """

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class EdgeGatedGraphConv(nn.Module):
    """
    residual gated graph convolution similar as in ALIGNN
    """

    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N, input_features)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge features (E, input_features)
        Returns:
            x_new: Updated node features
            edge_attr_new: Updated edge features
        """
        row, col = edge_index

        # Edge gating mechanisms
        # g.ndata["e_src"] = self.src_gate(node_feats) -> src_gate(x)[row]
        # g.ndata["e_dst"] = self.dst_gate(node_feats) -> dst_gate(x)[col]
        # g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        e_nodes = self.src_gate(x)[row] + self.dst_gate(x)[col]
        
        # m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)
        m = e_nodes + self.edge_gate(edge_attr)

        # g.edata["sigma"] = torch.sigmoid(m)
        sigma = torch.sigmoid(m)

        # g.ndata["Bh"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        Bh = self.dst_update(x)
        # Message: Bh[row] * sigma (using source node features to match DGL's u_mul_e if direction is u->v)
        # Wait, DGL's u_mul_e("Bh", "sigma", "m") takes "Bh" from u (source).
        # And fn.sum("m", "sum_sigma_h") aggregates to v (destination).
        # So we want Bh[row] * sigma, aggregated to col.
        messages = Bh[row] * sigma
        sum_sigma_h = scatter(messages, col, dim=0, reduce='sum', dim_size=x.size(0))

        # g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        sum_sigma = scatter(sigma, col, dim=0, reduce='sum', dim_size=x.size(0))

        # g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        h = sum_sigma_h / (sum_sigma + 1e-6)

        # x = self.src_update(node_feats) + g.ndata.pop("h")
        x_out = self.src_update(x) + h

        # node and edge updates
        # x = F.silu(self.bn_nodes(x))
        x_out = F.silu(self.bn_nodes(x_out))
        
        # y = F.silu(self.bn_edges(m))
        y_out = F.silu(self.bn_edges(m))

        # x = node_feats + x
        x_new = x + x_out
        
        # y = edge_feats + y
        edge_attr_new = edge_attr + y_out

        return x_new, edge_attr_new


class GraphConv(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Node and Edge updates for graph conv layer.

        x: node input features
        edge_attr: edge input features
        """
        # Edge-gated graph convolution update on crystal graph
        x, edge_attr = self.node_update(x, edge_index, edge_attr)

        return x, edge_attr


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


# nested graph convolution
class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
            self,
            x: torch.Tensor,
            edge_index_g: torch.Tensor,
            edge_attr_g: torch.Tensor,
            edge_index_lg: torch.Tensor,
            edge_attr_lg: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features (atoms)
        edge_attr_g: edge input features (bonds)
        edge_attr_lg: edge pair input features (angles)
        """
        # Edge-gated graph convolution update on crystal graph
        # Updates atoms(x) and bonds(y)
        # x, m = self.node_update(g, x, y)
        x, edge_attr_g = self.node_update(x, edge_index_g, edge_attr_g)

        # Edge-gated graph convolution update on line graph
        # Updates bonds(y) and angles(z)
        # Note: In line graph, the "nodes" are the bonds of the original graph.
        # So 'edge_attr_g' acts as node features for the line graph.
        # y, z = self.edge_update(lg, m, z)
        edge_attr_g, edge_attr_lg = self.edge_update(edge_attr_g, edge_index_lg, edge_attr_lg)

        return x, edge_attr_g, edge_attr_lg


class GraphEncoder(nn.Module):
    """
    Graph Encoder
    """

    def __init__(self, classification, atom_input_features, hidden_features, edge_input_features, embedding_features,
                 triplet_input_features, alignn_layers, gcn_layers, output_features):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.classification = classification

        self.atom_embedding = MLPLayer(
            atom_input_features, hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_input_features,
            ),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=triplet_input_features,
            ),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    hidden_features,
                    hidden_features,
                )
                for idx in range(alignn_layers)
            ]
        )

        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    hidden_features, hidden_features
                )
                for idx in range(gcn_layers)
            ]
        )

        self.fc = nn.Linear(hidden_features, output_features)
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, g, lg):
        """ALIGNN : start with `atom_features`.

        g: PyG Data object for crystal graph. 
           Expected attributes: atom_features, edge_index, r (bond vector), batch
        lg: PyG Data object for line graph.
           Expected attributes: edge_index, h (angle features)
        """
        
        # angle features (fixed)
        # z = self.angle_embedding(lg.edata.pop("h"))
        z = self.angle_embedding(lg.h)

        # initial node features: atom feature network...
        # x = g.ndata.pop("atom_features")
        x = g.atom_features
        x = self.atom_embedding(x)

        # initial bond features
        # bondlength = torch.norm(g.edata.pop("r"), dim=1)
        bondlength = torch.norm(g.r, dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(x, g.edge_index, y, lg.edge_index, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(x, g.edge_index, y)

        # norm-activation-pool-classify
        # h = self.readout(g, x) -> global_mean_pool(x, g.batch)
        h = global_mean_pool(x, g.batch)

        out = self.fc(h)
        if self.classification:
            out = self.softmax(out)
        
        return out
