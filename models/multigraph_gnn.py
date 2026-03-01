"""MultigraphGNN: Spatio-temporal GNN with relational graph convolution (RGCNConv).

Architecture mirrors MTGNN but replaces MixProp (dense matmul) with PyG's RGCNConv
for multi-relational graph convolution. Temporal processing uses the same
DilatedInception + gating mechanism.

Input/output format matches MTGNN exactly:
    Input:  (batch, 1, num_nodes, seq_len)
    Output: (batch, 1, num_nodes, 1)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RelationalGraphConv(nn.Module):
    """Wrapper around RGCNConv that handles the 4D tensor format used by MTGNN.

    MTGNN operates on (batch, channels, nodes, seq_len) tensors.
    RGCNConv expects (num_nodes_total, channels) with batched edge_index.
    This module reshapes per-timestep, applies RGCNConv, and reshapes back.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases=None):
        super().__init__()
        self.conv = RGCNConv(
            in_channels, out_channels, num_relations, num_bases=num_bases
        )

    def forward(self, X, edge_index, edge_type):
        """
        Args:
            X: (batch, channels, nodes, seq_len)
            edge_index: (2, E) — same graph for all batch elements
            edge_type: (E,)

        Returns:
            (batch, out_channels, nodes, seq_len)
        """
        B, C, N, T = X.shape

        # Build batched edge_index: replicate graph for each batch element
        offsets = torch.arange(B, device=X.device) * N  # (B,)
        # (B, 2, E) -> (2, B*E)
        batched_ei = (edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)
        batched_et = edge_type.repeat(B)

        outputs = []
        for t in range(T):
            h = X[:, :, :, t]           # (B, C, N)
            h = h.permute(0, 2, 1)      # (B, N, C)
            h = h.reshape(B * N, C)     # (B*N, C)
            h = self.conv(h, batched_ei, batched_et)  # (B*N, out_channels)
            h = h.reshape(B, N, -1)     # (B, N, out_channels)
            h = h.permute(0, 2, 1)      # (B, out_channels, N)
            outputs.append(h)

        return torch.stack(outputs, dim=3)  # (B, out_channels, N, T)


class MultigraphGNNLayer(nn.Module):
    """Single layer of MultigraphGNN: temporal conv + relational graph conv.

    Mirrors MTGNNLayer but with RGCNConv instead of MixProp.
    """

    def __init__(
        self,
        dilation_exponential,
        rf_size_i,
        kernel_size,
        j,
        residual_channels,
        conv_channels,
        skip_channels,
        kernel_set,
        new_dilation,
        layer_norm_affline,
        seq_length,
        receptive_field,
        dropout,
        num_nodes,
        num_relations,
        num_bases=None,
    ):
        super().__init__()
        self._dropout = dropout

        # Compute receptive field size at this layer
        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)

        # Temporal convolutions (identical to MTGNN)
        from models.mtgnn import DilatedInception
        self._filter_conv = DilatedInception(
            residual_channels, conv_channels,
            kernel_set=kernel_set, dilation_factor=new_dilation,
        )
        self._gate_conv = DilatedInception(
            residual_channels, conv_channels,
            kernel_set=kernel_set, dilation_factor=new_dilation,
        )

        # Relational graph convolution (replaces MixProp)
        self._rgcn = RelationalGraphConv(
            conv_channels, residual_channels, num_relations, num_bases=num_bases,
        )

        # Skip connection conv
        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                conv_channels, skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
            )
        else:
            self._skip_conv = nn.Conv2d(
                conv_channels, skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
            )

        # Layer norm
        if seq_length > receptive_field:
            norm_shape = (residual_channels, num_nodes, seq_length - rf_size_j + 1)
        else:
            norm_shape = (residual_channels, num_nodes, receptive_field - rf_size_j + 1)

        self._normalization = nn.LayerNorm(norm_shape, elementwise_affine=layer_norm_affline)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X, X_skip, edge_index, edge_type, training):
        """
        Args:
            X: (batch, residual_channels, nodes, seq_len)
            X_skip: (batch, skip_channels, nodes, T_skip)
            edge_index: (2, E)
            edge_type: (E,)
            training: bool

        Returns:
            X: (batch, residual_channels, nodes, seq_len')
            X_skip: (batch, skip_channels, nodes, T_skip)
        """
        X_residual = X

        # Temporal convolution with gating (same as MTGNN)
        X_filter = torch.tanh(self._filter_conv(X))
        X_gate = torch.sigmoid(self._gate_conv(X))
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)

        # Skip connection
        X_skip = self._skip_conv(X) + X_skip

        # Relational graph convolution (replaces MixProp)
        X = self._rgcn(X, edge_index, edge_type)

        # Residual connection
        X = X + X_residual[:, :, :, -X.size(3):]

        # Layer normalization
        X = self._normalization(X)

        return X, X_skip


class MultigraphGNN(nn.Module):
    """Spatio-temporal GNN with relational graph convolution for multigraph edges.

    Maintains the same I/O interface as MTGNN:
        Input:  (batch, in_dim, num_nodes, seq_len)
        Output: (batch, out_dim, num_nodes, 1)

    Uses RGCNConv to handle multiple edge types (piezo-piezo, piezo-pump, etc.)
    instead of the dense MixProp used in MTGNN.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int = 5,
        kernel_set: list = None,
        kernel_size: int = 2,
        dropout: float = 0.5,
        dilation_exponential: int = 2,
        conv_channels: int = 64,
        residual_channels: int = 64,
        skip_channels: int = 64,
        end_channels: int = 128,
        seq_length: int = 6,
        in_dim: int = 1,
        out_dim: int = 1,
        layers: int = 4,
        layer_norm_affline: bool = True,
        rgcn_num_bases: int = None,
        **kwargs,  # Accept and ignore extra config keys
    ):
        super().__init__()

        if kernel_set is None:
            kernel_set = [1, 2]

        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers

        self._mtgnn_layers = nn.ModuleList()

        # Compute receptive field (same logic as MTGNN)
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

        # Build layers
        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MultigraphGNNLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    num_nodes=num_nodes,
                    num_relations=num_relations,
                    num_bases=rgcn_num_bases,
                )
            )
            new_dilation *= dilation_exponential

        # Convolutions (same structure as MTGNN)
        self._start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))

        if seq_length > self._receptive_field:
            self._skip_conv_0 = nn.Conv2d(
                in_dim, skip_channels, kernel_size=(1, seq_length), bias=True
            )
            self._skip_conv_E = nn.Conv2d(
                residual_channels, skip_channels,
                kernel_size=(1, seq_length - self._receptive_field + 1), bias=True,
            )
        else:
            self._skip_conv_0 = nn.Conv2d(
                in_dim, skip_channels, kernel_size=(1, self._receptive_field), bias=True
            )
            self._skip_conv_E = nn.Conv2d(
                residual_channels, skip_channels, kernel_size=(1, 1), bias=True
            )

        self._end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self._end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X_in: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        edge_weight: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            X_in: (batch, in_dim, num_nodes, seq_len)
            edge_index: (2, E)
            edge_type: (E,)
            edge_weight: (E,) — currently unused, reserved for future use

        Returns:
            (batch, out_dim, num_nodes, 1)
        """
        seq_len = X_in.size(3)
        assert seq_len == self._seq_length, \
            f"Input seq_len {seq_len} != model seq_length {self._seq_length}"

        if self._seq_length < self._receptive_field:
            X_in = F.pad(X_in, (self._receptive_field - self._seq_length, 0, 0, 0))

        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )

        for layer in self._mtgnn_layers:
            X, X_skip = layer(X, X_skip, edge_index, edge_type, self.training)

        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        return X
