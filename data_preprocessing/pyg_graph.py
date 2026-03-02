import numpy as np
import torch
from pathlib import Path


def classify_edge_by_node_index(src, dst, num_piezo, num_pump, num_prec, num_evap, num_river):
    """Return integer edge type based on source and destination node indices.

    Node layout: [piezometers | pumps | precip | evap | rivers]
    Edge types:
        0 = piezo-piezo
        1 = piezo-pump (or pump-piezo)
        2 = piezo-precip (or precip-piezo)
        3 = piezo-evap (or evap-piezo)
        4 = piezo-river (or river-piezo)
    """
    pump_start = num_piezo
    prec_start = pump_start + num_pump
    evap_start = prec_start + num_prec
    river_start = evap_start + num_evap

    def node_type(idx):
        if idx < num_piezo:
            return 0  # piezo
        elif idx < prec_start:
            return 1  # pump
        elif idx < evap_start:
            return 2  # precip
        elif idx < river_start:
            return 3  # evap
        else:
            return 4  # river

    src_type = node_type(src)
    dst_type = node_type(dst)

    # Both piezometers
    if src_type == 0 and dst_type == 0:
        return 0
    # One is pump
    if src_type == 1 or dst_type == 1:
        return 1
    # One is precip
    if src_type == 2 or dst_type == 2:
        return 2
    # One is evap
    if src_type == 3 or dst_type == 3:
        return 3
    # One is river
    if src_type == 4 or dst_type == 4:
        return 4
    # Fallback (shouldn't happen with current graph construction)
    return 0


def dense_adj_to_pyg(adj_matrix, num_piezo, num_pump, num_prec, num_evap, num_river):
    """Convert a dense (N, N) adjacency matrix into PyG-compatible sparse format.

    Edge types are determined by node index ranges (robust to RF weight perturbation).

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Dense adjacency matrix.
    num_piezo, num_pump, num_prec, num_evap, num_river : int
        Number of nodes of each type.

    Returns
    -------
    dict with keys:
        'edge_index': torch.LongTensor of shape (2, E)
        'edge_type': torch.LongTensor of shape (E,)
        'edge_weight': torch.FloatTensor of shape (E,)
        'num_nodes': int
        'num_relations': int
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.numpy()

    rows, cols = np.nonzero(adj_matrix)
    weights = adj_matrix[rows, cols]

    edge_types = np.array([
        classify_edge_by_node_index(int(r), int(c), num_piezo, num_pump, num_prec, num_evap, num_river)
        for r, c in zip(rows, cols)
    ])

    num_nodes = adj_matrix.shape[0]
    num_relations = len(set(edge_types)) if len(edge_types) > 0 else 5

    return {
        'edge_index': torch.tensor(np.stack([rows, cols]), dtype=torch.long),
        'edge_type': torch.tensor(edge_types, dtype=torch.long),
        'edge_weight': torch.tensor(weights, dtype=torch.float32),
        'num_nodes': num_nodes,
        'num_relations': max(num_relations, 5),  # at least 5 relation types
    }


def save_pyg_graph(graph_dict, path):
    """Save PyG graph tensors to disk."""
    torch.save(graph_dict, path)


def load_pyg_graph(path):
    """Load PyG graph tensors from disk."""
    return torch.load(path, weights_only=False)
