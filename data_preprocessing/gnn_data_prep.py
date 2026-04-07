# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:09:58 2024

@author: cnmlt
"""
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime

from config import (
    PIEZO_METADATA, PUMP_METADATA, PUMP_DISTANCES, EVAP_METADATA,
    PREC_METADATA, RIVER_METADATA, PREPROCESSED_DIR, RANDOM_FOREST_TRAINING_DATA, GENERATED_GRAPHS,
    RF_TRAINED_ALL, RF_TRAINED_PIEZOS_ONLY, PIEZO_LAYER_INFORMATION,
    HYDRAULIC_RESISTANCE_DIR, PUMP_COHERENCE_WEIGHTS
)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_n_closest_pumps_indices(n, num_piezo, pump_distances_file=PUMP_DISTANCES, pump_columns=None):
    """
    Reads a CSV of piezometer-to-pump distances and returns, for each piezometer,
    the integer column indices of the n closest pumps, offset by num_piezo so they
    map to the correct node indices in the adjacency matrix.

    Parameters
    ----------
    n : int
        Number of closest pumps to find per piezometer.
    num_piezo : int
        Number of piezometers (used as offset for pump node indices).
    pump_distances_file : str
        Path to a CSV file with header and index column of shape (num_piezometers, num_pumps).
    pump_columns : list of str, optional
        Pump column names in adjacency-matrix order. If provided, the distance
        CSV columns are reordered to match before computing closest pumps.

    Returns
    -------
    np.ndarray
        Array of shape (num_piezometers, n) where row i contains the adjacency-matrix
        indices of the n closest pumps to piezometer i.
    """
    # load distances with header and index
    dist_df = pd.read_csv(pump_distances_file, header=0, index_col=0)

    # reorder columns to match adjacency pump ordering (from pump_metadata.csv)
    if pump_columns is not None:
        dist_df = dist_df[pump_columns]
    else:
        pump_meta = pd.read_csv(PUMP_METADATA)
        pump_order = pump_meta['Naam'].tolist()
        dist_df = dist_df[pump_order]

    dist_array = dist_df.values

    # argsort each row to get pump indices in ascending distance order
    sorted_pump_indices = np.argsort(dist_array, axis=1)

    # take the first n indices for each row, offset by num_piezo
    return sorted_pump_indices[:, :n] + num_piezo


def compute_log_pump_weights(
    num_piezo,
    n_pumps_connected=4,
    pump_distances_file=PUMP_DISTANCES,
    multiplier=1.0,
    w_min=0.1,
    w_max=0.3,
    R=10000,
    per_station_R=None,
):
    """
    Compute pump edge weights based on logarithmic drawdown decay (Thiem equation).

    Drawdown ∝ ln(R/r), where R is the radius of influence and r is the
    distance from the well.  Weights are scaled to [w_min, w_max].
    Piezometers beyond R are disconnected (weight = 0).

    Parameters
    ----------
    num_piezo : int
        Number of piezometers (used as offset for pump node indices).
    n_pumps_connected : int
        Number of closest pumps per piezometer (only those within R are kept).
    pump_distances_file : str or Path
        CSV with piezometer-to-pump distances (num_piezo × num_pumps, no header).
    multiplier : float
        Tunable scaling factor applied after the log mapping.
        multiplier=1.0 gives the full [w_min, w_max] range;
        <1 compresses towards w_min, >1 pushes towards w_max.
    w_min, w_max : float
        Output weight bounds (default 0.1–0.3).
    R : float
        Default radius of influence (m).  Piezometers beyond R get weight 0
        (disconnected).  Default 10 km.
    per_station_R : dict or None
        Per-pump radius override, e.g. {'Fikkersdries': 15000, 'Sijmons': 12000}.
        Pumps not listed use the default R.

    Returns
    -------
    weights_matrix : np.ndarray, shape (num_piezo, n_pumps)
        Full piezo × pump weight matrix. Zero = no connection.
    """
    dist_df = pd.read_csv(pump_distances_file, header=0, index_col=0)
    pump_names = dist_df.columns.tolist()
    dist_array = dist_df.values  # (P, 4)
    n_piezo, n_pumps = dist_array.shape

    # Build per-pump R array
    R_arr = np.full(n_pumps, float(R))
    if per_station_R:
        for j, pname in enumerate(pump_names):
            if pname in per_station_R:
                R_arr[j] = per_station_R[pname]

    # Thiem-style log influence for ALL pump-piezo pairs
    clamped_dist = np.clip(dist_array, 1.0, None)

    # ln(R_j/r_ij): positive when r < R_j, zero/negative when r >= R_j
    log_influence = np.log(R_arr[None, :] / clamped_dist)

    # Normalise to [0, 1] — use actual r_min across all within-range pairs
    within_range = dist_array < R_arr[None, :]
    r_min = clamped_dist[within_range].min() if within_range.any() else 1.0
    log_max = np.log(R_arr.max() / r_min)

    normed = np.clip(log_influence / log_max, 0, 1) if log_max > 0 else np.zeros_like(log_influence)
    w = w_min + (w_max - w_min) * normed * multiplier
    w = np.clip(w, w_min, w_max)

    # Zero out piezometers beyond their pump's R
    w[~within_range] = 0.0

    # Enforce n_pumps_connected limit: keep only n closest per piezo
    if n_pumps_connected < n_pumps:
        sorted_idx = np.argsort(dist_array, axis=1)
        for i in range(n_piezo):
            far_pumps = sorted_idx[i, n_pumps_connected:]
            w[i, far_pumps] = 0.0

    n_connected = (w > 0).sum()
    if per_station_R:
        r_str = ', '.join(f'{p}={int(r/1000)}km' for p, r in zip(pump_names, R_arr))
        print(f"  Thiem pump weights: per-station R=[{r_str}], "
              f"{n_connected}/{n_piezo * n_pumps} edges, "
              f"range=[{w[w > 0].min():.3f}, {w[w > 0].max():.3f}]")
    else:
        print(f"  Thiem pump weights: R={R/1000:.0f} km, "
              f"{n_connected}/{n_piezo * n_pumps} edges, "
              f"range=[{w[w > 0].min():.3f}, {w[w > 0].max():.3f}]")

    return w


def compute_coherence_pump_weights(
    num_piezo,
    num_pump,
    coherence_weights_file=PUMP_COHERENCE_WEIGHTS,
):
    """
    Load pre-computed coherence-based pump weights from CSV.

    The CSV has shape (num_piezo, num_pump) with index=piezometer names,
    columns=pump names.  Zero values mean no connection.

    Returns
    -------
    weights_matrix : np.ndarray, shape (num_piezo, num_pump)
        Per-edge weights (0 = disconnected).
    """
    df = pd.read_csv(coherence_weights_file, index_col=0)
    weights_matrix = df.values  # (num_piezo, num_pump)
    assert weights_matrix.shape == (num_piezo, num_pump), (
        f"Coherence weights shape {weights_matrix.shape} != ({num_piezo}, {num_pump})")
    n_connected = (weights_matrix > 0).sum()
    print(f"  Coherence pump weights: {n_connected}/{weights_matrix.size} edges, "
          f"range=[{weights_matrix[weights_matrix > 0].min():.3f}, "
          f"{weights_matrix[weights_matrix > 0].max():.3f}]")
    return weights_matrix


def build_same_layer_block(
    piezo_names: list,
    all_coords: np.ndarray,
    layer_csv: str,
    layer_column: str,
    n_piezo_connected: int = 3,
    weight: float = 0.1,
    weight_mode: str = 'fixed',  # 'fixed' or 'variable'
    rf_matrix: np.ndarray = None,
    rf_path: str = None,
    feature_importance_multiplier: float = 1.0
) -> np.ndarray:
    """
    Build a P×P adjacency block connecting each piezo only to its top peers
    within the same layer, using fixed weights or RF-based weights & selection.

    Parameters:
    - piezo_names: List of piezometer identifiers (length P)
    - all_coords:   Array of shape (N,2) of coordinates; not used here
    - layer_csv:    CSV path containing 'name' and layer_column
    - layer_column: Column name in CSV for layer labels
    - n_piezo_connected: Maximum neighbors per piezo
    - weight:       Static weight for fixed mode
    - weight_mode:  'fixed' or 'variable'
    - rf_matrix:   Preloaded P×P RF importance matrix (optional)
    - rf_path:     Path to a P×P RF matrix for loading if rf_matrix None
    - feature_importance_multiplier: Scaling factor for RF weights

    Returns:
    - Symmetric P×P adjacency block
    """
    # Load layer labels
    df = pd.read_csv(layer_csv).set_index("name")
    labels = df.loc[piezo_names, layer_column].fillna("MISSING").values
    P = len(piezo_names)

    # Prepare RF matrix if using variable weights
    if weight_mode == 'variable':
        if rf_matrix is not None:
            rf = rf_matrix.copy()
        elif rf_path is not None:
            rf = joblib.load(rf_path)
        else:
            raise ValueError("rf_matrix or rf_path must be provided for variable mode")

        if rf.shape != (P, P):
            raise ValueError(f"Expected RF matrix shape ({P},{P}), got {rf.shape}")
        rf *= feature_importance_multiplier

    # Initialize empty block
    block = np.zeros((P, P), dtype=float)

    # Build same-layer links
    for i in range(P):
        peers = [j for j in range(P) if j != i and labels[j] == labels[i]]
        if not peers:
            continue

        if weight_mode == 'variable':
            # select top-n peers by RF importance
            ranked = sorted(peers, key=lambda j: rf[i, j], reverse=True)
        else:
            # fixed mode: deterministic ordering of peers
            ranked = sorted(peers)

        selected = ranked[:n_piezo_connected]

        if weight_mode == 'variable':
            block[i, selected] = rf[i, selected]
        else:
            block[i, selected] = weight

    # Ensure symmetry
    return np.maximum(block, block.T)




def generate_adjacency_matrix(coordinates, threshold=0.5):
    """
    Generate adjacency matrix based on spatial coordinates and a given threshold.
    """
    num_nodes = len(coordinates)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = euclidean_distance(coordinates[i, 0], coordinates[i, 1], coordinates[j, 0], coordinates[j, 1])
            dist_matrix[i, j] = distance
            if distance < threshold:
                adj_matrix[i, j] = adj_matrix[j, i] = 1

    return adj_matrix


def generate_complex_adjacency_matrix(all_coords, num_piezo, num_pump, num_prec, num_evap, num_river, percentage=None, n_piezo_connected = 3, n_pumps_connected = 4):
    """
    Generate a complex adjacency matrix based on spatial coordinates and specific connectivity rules.

    Parameters:
    - all_coords: Numpy array of coordinates for all points (piezometers, pumps, precipitation, evaporation, river points).
    - num_piezo, num_pump, num_prec, num_evap, num_river: Number of each type of point.
    """
    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))

    # Compute the distance matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = euclidean_distance(all_coords[i, 0], all_coords[i, 1], all_coords[j, 0], all_coords[j, 1])

    # Always connect each piezometer to the 3 closest piezometers
    for i in range(num_piezo):
            # Connect to the 3 closest piezometers (excluding itself)
            piezo_indices = np.argsort(dist_matrix[i, :num_piezo])[2:2+n_piezo_connected]  # Skip the first index (itself)
            adj_matrix[i, piezo_indices] = 0.1
  
    # Determine the indices of nodes to connect to exhogenous variables
    if percentage is not None:
        # Calculate the number of nodes to process based on the percentage
        num_nodes_to_process = int(np.ceil(num_nodes * (percentage / 100.0)))
        # Randomly select the indices of the nodes to process
        selected_indices = random.sample(range(num_nodes), num_nodes_to_process)
    else:
        # If no percentage is given, process all nodes
        selected_indices = range(num_nodes)


    #Read pump distances
    pumps = pd.read_csv(PUMP_DISTANCES, header = 0, index_col = 0)
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES) 

    # Connectivity logic
    for i in selected_indices:
        if i < num_piezo:  # For piezometers
            # Connect to the 3 closest piezometers (excluding itself)
            piezo_indices = np.argsort(dist_matrix[i, :num_piezo])[2:5]  # Skip the first index (itself)
            adj_matrix[i, piezo_indices] = 0.1

            pump_indices = closest_pumps[i].tolist()
            for pump_col_idx in pump_indices:
                adj_matrix[i, pump_col_idx] = 0.2

            # Connect to the closest precipitation
            prec_index = num_piezo + num_pump + np.argmin(dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
            adj_matrix[i, prec_index] = 0.3

            # Connect to the closest evaporation
            evap_index = num_piezo + num_pump + num_prec + np.argmin(dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
            adj_matrix[i, evap_index] = 0.4

            # Connect to the two closest rivers
            river_indices = np.argsort(dist_matrix[i, -num_river:])[:2] + (num_nodes - num_river)
            adj_matrix[i, river_indices] = 0.5

    # Symmetrize the matrix for undirected connections
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    return adj_matrix


import pandas as pd
import numpy as np


def generate_layer_constrained_adjacency_matrix(
    all_coords, piezo_names, num_piezo, num_pump, num_prec, num_evap, num_river,
    layer_column="geolayer", percentage=None, n_piezo_connected=3, n_pumps_connected=4,
    exo_penalties=False, weight_mode='fixed'
):
    """
    Create adjacency matrix by connecting piezometers only if they are in the same layer
    (either 'geolayer' or 'regis_layer').

    Parameters
    ----------
    all_coords : ndarray
        Coordinates of all nodes (piezometers + exogenous).
    piezo_names : list
        List of piezometer names.
    layer_column : str
        Either 'geolayer' or 'regis_layer'.
    Returns
    -------
    adj_matrix : ndarray
        Symmetric adjacency matrix.
    """

    from preprocessing import euclidean_distance

    # Load geolayer/regis_layer info
    layer_df = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index("name")
    layer_labels = layer_df.loc[piezo_names, layer_column].fillna("MISSING").values

    #Read pump distances
    pumps = pd.read_csv(PUMP_DISTANCES, header = 0, index_col = 0)
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES) 

    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))


    aq2_indices_list = []
    if exo_penalties:
      for i in range(len(piezo_names)):
        if layer_df.loc[piezo_names[i], "geolayer"] == "aq2": aq2_indices_list.append(i)  #Now you have a list of indices of piezometer indices for aq2.


    rf_full = None
    if weight_mode == 'variable':
      rf_feature_importance = joblib.load(RF_TRAINED_PIEZOS_ONLY)
      P = rf_feature_importance.shape[0]
      if P != num_piezo:
        raise ValueError(
            f"Expected RF importances shape ({num_piezo},{num_piezo}), got {rf_feature_importance.shape}"
        )

      mask = rf_feature_importance > 0
      nonzero = rf_feature_importance[mask]
      min_w, max_w = nonzero.min(), nonzero.max()

      rf_scaled = np.zeros_like(rf_feature_importance)
      rf_scaled[mask] = (
          (rf_feature_importance[mask] - min_w)
          / (max_w - min_w)                 # now in [0,1]
          * (0.16 - 0.08)                    # now in [0, 0.08]
          + 0.08                            # now in [0.08, 0.16]
      )

      rf_full = rf_scaled


    # Compute distance matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = euclidean_distance(
                all_coords[i, 0], all_coords[i, 1], all_coords[j, 0], all_coords[j, 1]
            )


    # Connect piezometers only within same layer
    for i in range(num_piezo):
        same_layer = [
            j for j in range(num_piezo)
            if i != j and layer_labels[i] == layer_labels[j]
        ]
        if same_layer:
            nearest = sorted(same_layer, key=lambda j: dist_matrix[i, j])[:n_piezo_connected]
            if weight_mode == 'variable' and rf_full is not None:
              adj_matrix[i, nearest] = rf_full[i, nearest]
            else:
              adj_matrix[i, nearest] = 0.1
              


    # Connect piezometers to exogenous nodes
    selected_indices = (
        np.random.choice(num_nodes, int(np.ceil(num_nodes * (percentage / 100.0))), replace=False)
        if percentage is not None else range(num_nodes)
    )


    for i in selected_indices:
        if i < num_piezo:

            pump_indices = closest_pumps[i].tolist()
            prec_idx = num_piezo + num_pump + np.argmin(
                dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
            evap_idx = num_piezo + num_pump + num_prec + np.argmin(
                dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
            river_idxs = np.argsort(dist_matrix[i, -num_river:])[:2] + (num_nodes - num_river)

            adj_matrix[i, pump_indices] = 0.2
            
            #IF exo_penalties, remove the precip and evap connections from aq2 layer.
            if exo_penalties:
              if i not in aq2_indices_list:
                adj_matrix[i, prec_idx] = 0.3
                adj_matrix[i, evap_idx] = 0.4
            else:
              adj_matrix[i, prec_idx] = 0.3
              adj_matrix[i, evap_idx] = 0.4

            
            adj_matrix[i, river_idxs] = 0.5


    # Symmetrize the matrix for undirected connections
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    return adj_matrix
# TODO: WIP — Latent feature distance graph (Liu et al 2025)
# Uses euclidean distance + hydraulic conductivity + head for feature-based distance:
# dist(Vi,Vj) = sqrt(ΔX² + ΔY² + ΔZ² + ΔKx² + ΔKy² + ΔKz² + Δh²)
#
# def generate_latent_feature_distance_graph(
#         all_coords, piezo_names, num_piezo, num_pump, num_prec, num_evap, num_river,
#         layer_column="regis_layer", percentage=None, n_piezo_connected=3, n_pumps_connected=4
# ):
#     """
#     Parameters
#     ----------
#     all_coords : ndarray
#         Coordinates of all nodes (piezometers + exogenous).
#     piezo_names : list
#         List of piezometer names.
#     layer_column : str
#         Either 'geolayer' or 'regis_layer'.
#     Returns
#     -------
#     adj_matrix : ndarray
#         Symmetric adjacency matrix.
#     """
#     from preprocessing import euclidean_distance
#
#     layer_df = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index("name")
#     layer_labels = layer_df.loc[piezo_names, layer_column].fillna("MISSING").values
#
#     pumps = pd.read_csv(PUMP_DISTANCES, header=0, index_col=0)
#     closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES)
#
#     num_nodes = len(all_coords)
#     adj_matrix = np.zeros((num_nodes, num_nodes))
#     dist_matrix = np.zeros((num_nodes, num_nodes))
#
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             dist_matrix[i, j] = euclidean_distance(
#                 all_coords[i, 0], all_coords[i, 1], all_coords[j, 0], all_coords[j, 1]
#             )
#
#     for i in range(num_piezo):
#         same_layer = [
#             j for j in range(num_piezo)
#             if i != j and layer_labels[i] == layer_labels[j]
#         ]
#         if same_layer:
#             nearest = sorted(same_layer, key=lambda j: dist_matrix[i, j])[:n_piezo_connected]
#             adj_matrix[i, nearest] = 0.1
#
#     selected_indices = (
#         np.random.choice(num_nodes, int(np.ceil(num_nodes * (percentage / 100.0))), replace=False)
#         if percentage is not None else range(num_nodes)
#     )
#
#     for i in selected_indices:
#         if i < num_piezo:
#             pump_indices = closest_pumps[i].tolist()
#             prec_idx = num_piezo + num_pump + np.argmin(
#                 dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
#             evap_idx = num_piezo + num_pump + num_prec + np.argmin(
#                 dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
#             river_idxs = np.argsort(dist_matrix[i, -num_river:])[:2] + (num_nodes - num_river)
#
#             adj_matrix[i, pump_indices] = 0.2
#             adj_matrix[i, prec_idx] = 0.3
#             adj_matrix[i, evap_idx] = 0.4
#             adj_matrix[i, river_idxs] = 0.5
#
#     adj_matrix = adj_matrix + adj_matrix.T
#     return adj_matrix

def generate_rf_adjacency_fixed(
    piezo_columns: list,
    all_coords: np.ndarray,
    num_piezo: int,
    num_pump: int,
    num_prec: int,
    num_evap: int,
    num_river: int,
    n_top_connections: int = 3,
    n_pumps_connected: int = 4,
    layer_constrain: bool = False
) -> np.ndarray:
    """
    Generate adjacency matrix using RF feature importance with fixed edge weights.
    Piezo→piezo edges: top-n peers (fixed weight 0.1), optionally constrained to same geolayer.
    Exogenous edges use fixed weights: pumps (0.2), precipitation (0.3), evaporation (0.4), rivers (0.5).
    """
    num_nodes = all_coords.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Load RF importance matrix (P×P)
    rf_feature_importance = joblib.load(RF_TRAINED_PIEZOS_ONLY)
    P = rf_feature_importance.shape[0]
    if P != num_piezo:
        raise ValueError(
            f"Expected RF importances shape ({num_piezo},{num_piezo}), got {rf_feature_importance.shape}"
        )

    # Load layer labels if constraining to same layer
    piezo_labels = None
    if layer_constrain:
        df_layers = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index('name')
        piezo_labels = df_layers.loc[piezo_columns, "geolayer"].fillna('MISSING').values

    # Read closest pumps
    closest_pumps = get_n_closest_pumps_indices(
        n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES
    )

    # 1) Piezo→piezo: top-n importances (optionally among same-layer peers)
    for i in range(num_piezo):
        if layer_constrain:
            candidates = [j for j in range(num_piezo)
                          if j != i and piezo_labels[j] == piezo_labels[i]]
            if not candidates:
                continue
            weights = rf_feature_importance[i, candidates]
            topk_rel = np.argsort(weights)[-n_top_connections:][::-1]
            selected = [candidates[idx] for idx in topk_rel]
        else:
            row = rf_feature_importance[i]
            selected = np.argsort(row)[-n_top_connections:][::-1].tolist()
        for j in selected:
            adj_matrix[i, j] = 0.1

    # 2) Exogenous connections
    for i in range(num_piezo):
        pump_idxs = closest_pumps[i].tolist()
        adj_matrix[i, pump_idxs] = 0.2

        dist = np.linalg.norm(all_coords[i] - all_coords, axis=1)

        if num_prec > 0:
            prec_start = num_piezo + num_pump
            prec_idx = prec_start + np.argmin(dist[prec_start:prec_start + num_prec])
            adj_matrix[i, prec_idx] = 0.3

        if num_evap > 0:
            evap_start = num_piezo + num_pump + num_prec
            evap_idx = evap_start + np.argmin(dist[evap_start:evap_start + num_evap])
            adj_matrix[i, evap_idx] = 0.4

        if num_river > 0:
            river_start = num_nodes - num_river
            river_slice = dist[river_start:]
            two = np.argsort(river_slice)[:2] + river_start
            adj_matrix[i, two] = 0.5

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix


def generate_rf_cutoff_adjacency(
    all_coords: np.ndarray,
    num_piezo: int,
    num_pump: int,
    num_prec: int,
    num_evap: int,
    num_river: int,
    rf_config: dict,
) -> np.ndarray:
    """
    RF adjacency with importance-threshold cutoff.

    Piezo-piezo edges are determined by RF importance (from RF_TRAINED_ALL)
    with a cutoff threshold and minimum-connections guarantee.
    Pump and river connections use spatial distance (the RF matrix may have
    been trained with a different node count, and RF pump/river importances
    are negligible at medium-to-sparse cutoffs).
    Precip/evap use fixed weights (0.3 / 0.4) via spatial distance.

    Parameters
    ----------
    all_coords : ndarray (N, 2)
        XY coordinates for all N nodes.
    num_piezo, num_pump, num_prec, num_evap, num_river : int
        Node counts by type.
    rf_config : dict
        'cutoff'          : float  – min RF importance to create an edge
        'min_connections' : int    – guaranteed min piezo-piezo connections (default 3)
        'n_pumps_connected' : int  – number of closest pumps per piezo (default 4)
        'n_rivers_connected': int  – number of closest rivers per piezo (default 2)
        'pump_weight'     : float  – weight for pump edges (default 1.0)
        'river_weight'    : float  – weight for river edges (default 1.0)
    """
    cutoff = rf_config['cutoff']
    min_connections = rf_config.get('min_connections', 3)
    n_pumps_connected = rf_config.get('n_pumps_connected', 4)
    n_rivers_connected = rf_config.get('n_rivers_connected', 2)
    pump_weight = rf_config.get('pump_weight', 1.0)
    river_weight = rf_config.get('river_weight', 1.0)
    num_nodes = len(all_coords)

    # ── 1. Load RF importance matrix (piezo-only block) ──
    raw_rf = joblib.load(RF_TRAINED_ALL)
    if isinstance(raw_rf, pd.DataFrame):
        rf_full = raw_rf.values.astype(float)
    elif isinstance(raw_rf, np.ndarray):
        rf_full = raw_rf.astype(float)
    else:
        rf_full = np.array(raw_rf, dtype=float)

    print(f"  RF importance matrix shape: {rf_full.shape}")

    adj = np.zeros((num_nodes, num_nodes), dtype=float)

    # ── 2. Piezo-piezo connections (cutoff + min_connections guarantee) ──
    piezo_conn_counts = []
    for i in range(num_piezo):
        row = rf_full[i, :num_piezo].copy()
        row[i] = 0  # no self-loops
        mask = row >= cutoff
        if mask.sum() < min_connections:
            topk = np.argsort(row)[-min_connections:]
            mask[topk] = True
        adj[i, np.where(mask)[0]] = 1.0
        piezo_conn_counts.append(int(mask.sum()))

    # ── 3. Pump connections (spatial distance) ──
    closest_pumps = get_n_closest_pumps_indices(
        n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES
    )
    for i in range(num_piezo):
        adj[i, closest_pumps[i]] = pump_weight

    # ── 4. River connections (spatial distance, N closest) ──
    dist_matrix = pairwise_distances(all_coords)
    river_adj_start = num_nodes - num_river
    if num_river > 0:
        for i in range(num_piezo):
            riv_dists = dist_matrix[i, river_adj_start:river_adj_start + num_river]
            nearest = np.argsort(riv_dists)[:n_rivers_connected] + river_adj_start
            adj[i, nearest] = river_weight

    # ── 5. Precipitation and evaporation (fixed weights, spatial distance) ──
    prec_start = num_piezo + num_pump
    evap_start = prec_start + num_prec
    for i in range(num_piezo):
        if num_prec > 0:
            prec_idx = prec_start + np.argmin(dist_matrix[i, prec_start:prec_start + num_prec])
            adj[i, prec_idx] = 0.3
        if num_evap > 0:
            evap_idx = evap_start + np.argmin(dist_matrix[i, evap_start:evap_start + num_evap])
            adj[i, evap_idx] = 0.4

    # ── 6. Symmetrise ──
    adj = np.maximum(adj, adj.T)

    # ── 7. Stats ──
    piezo_edges = np.count_nonzero(adj[:num_piezo, :num_piezo]) // 2
    edges_per_node = np.count_nonzero(adj[:num_piezo, :num_piezo], axis=1)
    total_edges = np.count_nonzero(adj) // 2
    print(f"  RF-cutoff graph (cutoff={cutoff}, min_conn={min_connections}):")
    print(f"    Piezo-piezo edges: {piezo_edges}")
    print(f"    Edges/piezo: min={edges_per_node.min()}, mean={edges_per_node.mean():.1f}, max={edges_per_node.max()}")
    print(f"    Pump edges/piezo: {n_pumps_connected} (spatial, weight={pump_weight})")
    print(f"    River edges/piezo: {min(n_rivers_connected, num_river)} (spatial, weight={river_weight})")
    print(f"    Total edges (inc. exo): {total_edges}")

    return adj

def generate_rf_adjacency_variable(
        piezo_columns: list,
        all_coords: np.ndarray,
        num_piezo: int,
        num_pump: int,
        num_prec: int,
        num_evap: int,
        num_river: int,
        n_top_connections: int = 3,
        feature_importance_multiplier: float = 1.0,
        n_pumps_connected: int = 4,
        rf_weight_min: float = 0.08,
        rf_weight_max: float = 0.2,
        layer_constrain: bool = False
) -> np.ndarray:
    """
    Generate adjacency matrix using RF feature importance with variable (scaled) edge weights.
    Piezo→piezo edges: top-n peers by RF importance, weighted by scaled RF values.
    Optionally constrained to same geolayer when layer_constrain=True.
    Exogenous edges use fixed weights: pumps (0.2), precipitation (0.3), evaporation (0.4), rivers (0.5).
    """
    # Load the FULL RF matrix (N×N)
    raw_rf = joblib.load(RF_TRAINED_ALL)
    if isinstance(raw_rf, pd.DataFrame):
        rf_full = raw_rf.values.astype(float)
    elif isinstance(raw_rf, np.ndarray):
        rf_full = raw_rf.astype(float)
    else:
        rf_full = np.array(raw_rf, dtype=float)

    # Actual node count (may differ from RF matrix if nodes were dropped, e.g. daily resampling)
    num_nodes = num_piezo + num_pump + num_prec + num_evap + num_river

    # Scale RF weights to configured range [rf_weight_min, rf_weight_max]
    nonzero = rf_full[rf_full > 0]
    if nonzero.size == 0:
        raise ValueError("No nonzero importances found in RF matrix.")
    min_w, max_w = nonzero.min(), nonzero.max()
    rf_scaled = np.zeros_like(rf_full)
    rf_scaled[rf_full > 0] = (
        (rf_full[rf_full > 0] - min_w)
        / (max_w - min_w)
        * (rf_weight_max - rf_weight_min)
        + rf_weight_min
    )
    rf_full = rf_scaled
    print(f"RF piezo weights scaled to [{rf_weight_min}, {rf_weight_max}]")

    # Load layer labels if constraining to same layer
    piezo_labels = None
    if layer_constrain:
        df_layers = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index('name')
        piezo_labels = df_layers.loc[piezo_columns, "geolayer"].fillna('MISSING').values

    # Prepare adjacency — sized to actual node count, not RF matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    closest_pumps = get_n_closest_pumps_indices(
        n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES
    )

    # Piezo→piezo: top-k (optionally within same layer)
    for i in range(num_piezo):
        if layer_constrain:
            candidates = [j for j in range(num_piezo)
                          if j != i and piezo_labels[j] == piezo_labels[i]]
            if not candidates:
                continue
            weights = rf_full[i, candidates]
            topk_idx = np.argsort(weights)[-n_top_connections:][::-1]
            selected = [candidates[k] for k in topk_idx]
            adj_matrix[i, selected] = weights[topk_idx] * feature_importance_multiplier
        else:
            row = rf_full[i, :num_piezo].copy()
            row[i] = 0  # no self-loops
            topk = np.argsort(row)[-n_top_connections:][::-1]
            adj_matrix[i, topk] = row[topk] * feature_importance_multiplier

    # Exogenous connections with fixed weights
    start_pump = num_piezo
    start_prec = start_pump + num_pump
    start_evap = start_prec + num_prec
    start_river = start_evap + num_evap

    for i in range(num_piezo):
        # Pumps
        pump_idxs = np.array(closest_pumps[i], dtype=int)
        adj_matrix[i, pump_idxs] = 0.2

        dist = np.linalg.norm(all_coords[i] - all_coords, axis=1)

        # Precipitation
        if num_prec > 0:
            prec_idx = start_prec + np.argmin(dist[start_prec:start_prec + num_prec])
            adj_matrix[i, prec_idx] = 0.3

        # Evaporation
        if num_evap > 0:
            evap_idx = start_evap + np.argmin(dist[start_evap:start_evap + num_evap])
            adj_matrix[i, evap_idx] = 0.4

        # Rivers (two closest)
        if num_river > 0:
            river_slice = dist[start_river:start_river + num_river]
            two_rivs = np.argsort(river_slice)[:2] + start_river
            adj_matrix[i, two_rivs] = 0.5

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix


def generate_rf_full_vim_matrix(
        piezo_columns: list,
        all_coords: np.ndarray,
        num_piezo: int,
        num_pump: int,
        num_prec: int,
        num_evap: int,
        num_river: int,
        vim_min: float = 0.01,
        n_top_connections: int = None,
        n_min_connections: int = 3,
        n_pumps_connected: int = 4,
        feature_importance_multiplier: float = 1.0,
        rf_weight_min: float = 0.08,
        rf_weight_max: float = 0.2
) -> np.ndarray:
    """
    Full RF adjacency: keep ALL piezo-piezo connections where
    RF importance >= vim_min. If a node ends up with fewer than
    n_min_connections, its top-n peers by importance are added regardless
    of threshold. Optionally also limit to top-N per node.
    Exogenous edges use fixed weights: pumps (0.2), precipitation (0.3), evaporation (0.4), rivers (0.5).
    """
    # Load the FULL RF matrix (N×N)
    raw_rf = joblib.load(RF_TRAINED_ALL)
    if isinstance(raw_rf, pd.DataFrame):
        rf_full = raw_rf.values.astype(float)
    elif isinstance(raw_rf, np.ndarray):
        rf_full = raw_rf.astype(float)
    else:
        rf_full = np.array(raw_rf, dtype=float)

    # Actual node count (may differ from RF matrix if nodes were dropped)
    num_nodes = num_piezo + num_pump + num_prec + num_evap + num_river

    print(f"VIM threshold: {vim_min}, top-N cap: {n_top_connections}, min connections: {n_min_connections}")

    # Prepare adjacency — sized to actual node count, not RF matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=float)
    dmat = pairwise_distances(all_coords)
    closest_pumps = get_n_closest_pumps_indices(
        n_pumps_connected, num_piezo, pump_distances_file=PUMP_DISTANCES
    )

    # Piezo→piezo: apply VIM threshold on RAW importances, then scale survivors
    conn_counts = []
    for i in range(num_piezo):
        row = rf_full[i, :num_piezo].copy()
        row[i] = 0  # no self-loops
        mask = row >= vim_min
        # Guarantee minimum connectivity: if threshold leaves too few,
        # add top-n peers by importance
        if mask.sum() < n_min_connections:
            topk = np.argsort(row)[-n_min_connections:]
            mask[topk] = True
        if n_top_connections is not None:
            candidates = np.where(mask)[0]
            if len(candidates) > n_top_connections:
                topk = candidates[np.argsort(row[candidates])[-n_top_connections:]]
                mask[:] = False
                mask[topk] = True
        adj[i, np.where(mask)] = row[np.where(mask)]
        conn_counts.append(int(mask.sum()))

    # Scale surviving edge weights to [rf_weight_min, rf_weight_max]
    nonzero = adj[adj > 0]
    if nonzero.size > 0:
        min_w, max_w = nonzero.min(), nonzero.max()
        adj[adj > 0] = (
            (adj[adj > 0] - min_w)
            / (max_w - min_w)
            * (rf_weight_max - rf_weight_min)
            + rf_weight_min
        )
    print(f"RF piezo weights scaled to [{rf_weight_min}, {rf_weight_max}]")

    print(f"Piezo-piezo connections per node: min={min(conn_counts)}, "
          f"max={max(conn_counts)}, mean={np.mean(conn_counts):.1f}")

    # Exogenous connections with fixed weights
    start_pump = num_piezo
    start_prec = start_pump + num_pump
    start_evap = start_prec + num_prec
    start_river = start_evap + num_evap

    for i in range(num_piezo):
        # Pumps
        pump_idxs = np.array(closest_pumps[i], dtype=int)
        adj[i, pump_idxs] = 0.2

        # Precipitation
        if num_prec > 0:
            prec_idx = start_prec + np.argmin(dmat[i, start_prec:start_prec + num_prec])
            adj[i, prec_idx] = 0.3

        # Evaporation
        if num_evap > 0:
            evap_idx = start_evap + np.argmin(dmat[i, start_evap:start_evap + num_evap])
            adj[i, evap_idx] = 0.4

        # Rivers (two closest)
        if num_river > 0:
            riv_slice = dmat[i, start_river:start_river + num_river]
            two_rivs = np.argsort(riv_slice)[:2] + start_river
            adj[i, two_rivs] = 0.5

    adj = np.maximum(adj, adj.T)
    return adj


import numpy as np
from typing import Dict, List, Tuple


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def create_static_features(all_x, all_y, all_z, all_type):
    """
    Prepare static features for each node, such as spatial coordinates and categorical types.
    """
    # Normalize features
    x_normalized = normalize(all_x)
    y_normalized = normalize(all_y)
    z_normalized = normalize(all_z)

    # Combine into a single tensor
    static_features = torch.tensor(np.column_stack((x_normalized, y_normalized, z_normalized, all_type)), dtype=torch.float)

    return static_features

import os

def load_and_concatenate_metadata(piezo_metadata_path, pump_metadata_path, evap_metadata_path, prec_metadata_path, river_metadata_path, df_piezo_columns, pump_columns, locations_no_missing):

    # Load metadata for piezometers
    piezo_metadata = pd.read_csv(piezo_metadata_path)
    filtered_metadata = piezo_metadata[piezo_metadata['name'].isin(df_piezo_columns)]
    print(filtered_metadata.shape)
    filtered_metadata = filtered_metadata.set_index('name').reindex(df_piezo_columns).reset_index()
    piezo_z = (filtered_metadata['top_filter'] + filtered_metadata['bottom_filter'])/2


    # Load pump locations
    pump_metadata = pd.read_csv(pump_metadata_path)
    pump_metadata = pump_metadata[pump_metadata['Naam'].isin(pump_columns)]
    pump_metadata = pump_metadata.set_index('Naam').reindex(pump_columns).reset_index()

    # Load precipitation and evaporation locations
    evap_metadata = pd.read_csv(evap_metadata_path)
    prec_metadata = pd.read_csv(prec_metadata_path)

    # Load river locations
    river_metadata = pd.read_csv(river_metadata_path)
    river_metadata = river_metadata[river_metadata['name'].isin(locations_no_missing)]

    # build the name‐lists in the same order as the coords
    piezo_names = filtered_metadata['name'].tolist()
    pump_names  = pump_metadata['Naam'].tolist()
    prec_names  = prec_metadata['name'].tolist()
    evap_names  = evap_metadata['name'].tolist()
    river_names = river_metadata['name'].tolist()

    node_names = piezo_names + pump_names + prec_names + evap_names + river_names

    # Concatenate all locations and counts
    return (
        np.concatenate([filtered_metadata.iloc[:, 1].to_numpy(), pump_metadata['Xcoor'].to_numpy(), prec_metadata['x'].to_numpy(), evap_metadata['x'].to_numpy(), river_metadata['x'].to_numpy()]),
        np.concatenate([filtered_metadata.iloc[:, 2].to_numpy(), pump_metadata['Ycoor'].to_numpy(), prec_metadata['y'].to_numpy(), evap_metadata['y'].to_numpy(), river_metadata['y'].to_numpy()]),
        np.concatenate([piezo_z.to_numpy(), np.zeros_like(pump_metadata['Xcoor'].to_numpy()), np.zeros_like(prec_metadata['x'].to_numpy()), np.zeros_like(evap_metadata['x'].to_numpy()), np.zeros_like(river_metadata['x'].to_numpy())]),
        np.concatenate([np.ones_like(filtered_metadata.iloc[:, 3].to_numpy()), 2*np.ones_like(pump_metadata['Xcoor'].to_numpy()), 3*np.ones_like(prec_metadata['x'].to_numpy()), 4*np.ones_like(evap_metadata['x'].to_numpy()), 5*np.ones_like(river_metadata['x'].to_numpy())]),
        len(filtered_metadata.iloc[:, 3]),  # num_piezo
        4,  # num_pump
        len(prec_metadata['x']),  # num_prec
        len(evap_metadata['x']),  # num_evap
        len(river_metadata['x']), # num_river
        node_names  
    )


def generate_shortest_path_adjacency(
    all_coords, piezo_names, num_piezo, num_pump, num_prec, num_evap, num_river,
    resistance_source='regis',
    n_piezo_connected=3,
    sp_min_sensitivity=0.0,
    sp_min_connections=3,
    piezo_weight_range=(0.08, 0.2),
    pump_weight_range=(0.15, 0.25),
    river_weight_range=(0.4, 0.6),
    n_pumps_connected=4,
    n_rivers_connected=2,
    include_pumps=True,
    include_rivers=True,
    use_hydraulic_exo=False,
):
    """
    Build adjacency matrix from pre-computed shortest-path resistance.

    Loads a 209x209 resistance matrix (200 piezo + 4 pump + 5 river) and
    builds an adjacency matrix with configurable connectivity and weights.

    Parameters
    ----------
    resistance_source : str
        Which resistance .npy to load: 'regis' (full REGIS II K values) or
        'binary' (fixed K per layer type: aquifer vs aquitard).
    n_piezo_connected : int
        Number of top-N piezo-piezo connections per node.
    sp_min_sensitivity : float
        Minimum connectivity (1/resistance) to form an edge. Edges below
        this threshold are dropped unless sp_min_connections requires them.
    sp_min_connections : int
        Guaranteed minimum piezo-piezo connections per node. If the
        threshold yields fewer, top-N by connectivity are used instead.
    piezo_weight_range : tuple (float, float)
        (min, max) weight range for piezo-piezo edges (scaled by connectivity).
    pump_weight_range : tuple (float, float)
        (min, max) weight range for piezo-pump edges.
    river_weight_range : tuple (float, float)
        (min, max) weight range for piezo-river edges.
    n_pumps_connected : int
        Number of pumps to connect per piezometer.
    n_rivers_connected : int
        Number of rivers to connect per piezometer.
    include_pumps : bool
        Whether to include pump connections (if False, pump rows/cols are zero).
    include_rivers : bool
        Whether to include river connections (if False, river rows/cols are zero).
    use_hydraulic_exo : bool
        If True, use hydraulic resistance (from the full 209x209 matrix) for
        pump and river connections instead of distance-based assignment.
    """
    # ── Load resistance matrix ──
    res_file = HYDRAULIC_RESISTANCE_DIR / f'resistance_{resistance_source}.npy'
    res_full = np.load(res_file)

    # Determine matrix format
    if res_full.shape[0] == num_piezo:
        # Piezo-only matrix (old GeoTOP format), ordered by metadata CSV
        meta = pd.read_csv(PIEZO_METADATA)
        notebook_order = meta['name'].tolist()
        nb_name_to_idx = {n: i for i, n in enumerate(notebook_order)}
        reorder = [nb_name_to_idx[n] for n in piezo_names]
        res_piezo = res_full[np.ix_(reorder, reorder)]
        res_pump = None
        res_river = None
    elif res_full.shape[0] > num_piezo:
        # Full infrastructure matrix (209x209): piezo + pump + river in pipeline order
        res_piezo = res_full[:num_piezo, :num_piezo]
        pump_start = num_piezo
        river_start = num_piezo + num_pump
        res_pump = res_full[:num_piezo, pump_start:pump_start + num_pump]
        res_river = res_full[:num_piezo, river_start:river_start + num_river]
    else:
        raise ValueError(f"Unexpected resistance matrix shape {res_full.shape}")

    # ── Piezo-piezo connectivity ──
    connectivity = np.where(res_piezo > 0, 1.0 / res_piezo, 0.0)
    np.fill_diagonal(connectivity, 0)

    piezo_block = np.zeros((num_piezo, num_piezo))
    pw_min, pw_max = piezo_weight_range

    for i in range(num_piezo):
        row = connectivity[i]
        if sp_min_sensitivity > 0:
            mask = row >= sp_min_sensitivity
            if mask.sum() < sp_min_connections:
                topk = np.argsort(row)[-sp_min_connections:]
                mask[topk] = True
            selected = np.where(mask)[0]
        else:
            # Top-N mode
            ranked = np.argsort(row)[-n_piezo_connected:]
            selected = ranked[row[ranked] > 0]

        if len(selected) > 0:
            piezo_block[i, selected] = row[selected]

    piezo_block = np.maximum(piezo_block, piezo_block.T)

    # Scale piezo weights to range
    pos = piezo_block > 0
    if pos.any():
        vals = piezo_block[pos]
        scaled = pw_min + (vals - vals.min()) / (vals.max() - vals.min() + 1e-10) * (pw_max - pw_min)
        piezo_block[pos] = scaled

    # ── Build full adjacency ──
    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[:num_piezo, :num_piezo] = piezo_block

    dist_matrix = pairwise_distances(all_coords)

    # ── Pump connections ──
    if include_pumps and num_pump > 0:
        pmp_min, pmp_max = pump_weight_range
        if use_hydraulic_exo and res_pump is not None:
            pump_conn = np.where(res_pump > 0, 1.0 / res_pump, 0.0)
            for i in range(num_piezo):
                ranked = np.argsort(pump_conn[i])[-n_pumps_connected:]
                sel = ranked[pump_conn[i, ranked] > 0]
                if len(sel) > 0:
                    vals = pump_conn[i, sel]
                    scaled = pmp_min + (vals - vals.min()) / (vals.max() - vals.min() + 1e-10) * (pmp_max - pmp_min)
                    adj_matrix[i, num_piezo + sel] = scaled
        else:
            closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, num_piezo)
            for i in range(num_piezo):
                adj_matrix[i, closest_pumps[i]] = (pmp_min + pmp_max) / 2

    # ── Precipitation connections (always distance-based) ──
    for i in range(num_piezo):
        prec_idx = num_piezo + num_pump + np.argmin(
            dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
        adj_matrix[i, prec_idx] = 0.3

    # ── Evaporation connections (always distance-based) ──
    for i in range(num_piezo):
        evap_idx = num_piezo + num_pump + num_prec + np.argmin(
            dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
        adj_matrix[i, evap_idx] = 0.4

    # ── River connections ──
    if include_rivers and num_river > 0:
        riv_min, riv_max = river_weight_range
        if use_hydraulic_exo and res_river is not None:
            river_conn = np.where(res_river > 0, 1.0 / res_river, 0.0)
            for i in range(num_piezo):
                ranked = np.argsort(river_conn[i])[-n_rivers_connected:]
                sel = ranked[river_conn[i, ranked] > 0]
                if len(sel) > 0:
                    vals = river_conn[i, sel]
                    scaled = riv_min + (vals - vals.min()) / (vals.max() - vals.min() + 1e-10) * (riv_max - riv_min)
                    river_offset = num_nodes - num_river
                    adj_matrix[i, river_offset + sel] = scaled
        else:
            river_offset = num_nodes - num_river
            for i in range(num_piezo):
                river_dists = dist_matrix[i, river_offset:river_offset + num_river]
                nearest = np.argsort(river_dists)[:n_rivers_connected]
                adj_matrix[i, river_offset + nearest] = (riv_min + riv_max) / 2

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix


def generate_feature_distance_adjacency(
    all_coords, num_piezo, num_pump, num_prec, num_evap, num_river, fd_config,
):
    """
    Build adjacency matrix from a pre-computed 7-D feature-distance matrix
    (Liang et al. 2025).

    Loads the 209×209 distance matrix produced by extract_node_k_values.py,
    applies a radius cutoff, and sets edge weights = 1/distance.
    Precip/evap nodes get standard fixed weights.

    Parameters
    ----------
    all_coords : ndarray (N, 2)
        XY coordinates for all nodes (for precip/evap assignment).
    num_piezo, num_pump, num_prec, num_evap, num_river : int
        Node counts by type.
    fd_config : dict
        Must contain 'radius'; optional 'min_connections' (default 3).
        Optional 'pump_weight' (default 1.0), 'river_weight' (default 1.0).
        Optional 'weight_max' (default None = binary 1.0).
            When set, edge weights are scaled by inverse distance:
            w = weight_max * (1 - dist/radius), clamped to [0.01*weight_max, weight_max].
    """
    from config import FEATURE_DISTANCE_7D

    radius = fd_config['radius']
    min_connections = fd_config.get('min_connections', 3)
    pump_weight = fd_config.get('pump_weight', 1.0)
    river_weight = fd_config.get('river_weight', 1.0)
    weight_max = fd_config.get('weight_max', None)  # None = binary 1.0
    num_nodes = len(all_coords)

    # ── 1. Load pre-computed 7-D distance matrix (209×209) ──
    dist_7d = np.load(FEATURE_DISTANCE_7D)
    n_subsurface = dist_7d.shape[0]  # piezo + pump + river = 209

    # ── 2. Build subsurface adjacency block ──
    sub_adj = np.zeros((n_subsurface, n_subsurface))

    for i in range(n_subsurface):
        dists = dist_7d[i].copy()
        dists[i] = np.inf  # exclude self

        # Nodes within radius
        within_radius = np.where(dists <= radius)[0]

        if len(within_radius) < min_connections:
            # Guarantee minimum connections: pick closest N
            closest = np.argsort(dists)[:min_connections]
            within_radius = closest

        if weight_max is not None:
            # Distance-scaled weights: linear decay from weight_max at dist=0
            # to ~0 at dist=radius; forced connections beyond radius get floor
            floor = 0.01 * weight_max
            weights = weight_max * (1.0 - dists[within_radius] / radius)
            weights = np.clip(weights, floor, weight_max)
            sub_adj[i, within_radius] = weights
        else:
            sub_adj[i, within_radius] = 1.0  # unweighted (Liang et al. 2025)

    # Symmetrise
    sub_adj = np.maximum(sub_adj, sub_adj.T)

    # ── 3. Map subsurface block into full adjacency matrix ──
    # Subsurface order: piezo(0..P-1), pump(P..P+Pu-1), river(P+Pu..P+Pu+R-1)
    # Full adj order:   piezo(0..P-1), pump(P..P+Pu-1), prec, evap, river(N-R..N-1)
    # If daily data drops a river station, trim sub_adj to match num_river
    n_sub_river = n_subsurface - num_piezo - num_pump
    if n_sub_river > num_river:
        keep = list(range(num_piezo + num_pump)) + list(range(num_piezo + num_pump, num_piezo + num_pump + num_river))
        sub_adj = sub_adj[np.ix_(keep, keep)]

    adj_matrix = np.zeros((num_nodes, num_nodes))

    pump_sub_start = num_piezo
    pump_sub_end = num_piezo + num_pump
    river_sub_start = num_piezo + num_pump
    river_adj_start = num_nodes - num_river

    # Piezo-piezo
    adj_matrix[:num_piezo, :num_piezo] = sub_adj[:num_piezo, :num_piezo]

    # Piezo-pump (scaled by pump_weight)
    adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump] = sub_adj[:num_piezo, pump_sub_start:pump_sub_end] * pump_weight
    adj_matrix[num_piezo:num_piezo + num_pump, :num_piezo] = sub_adj[pump_sub_start:pump_sub_end, :num_piezo] * pump_weight

    # Piezo-river (scaled by river_weight)
    adj_matrix[:num_piezo, river_adj_start:] = sub_adj[:num_piezo, river_sub_start:] * river_weight
    adj_matrix[river_adj_start:, :num_piezo] = sub_adj[river_sub_start:, :num_piezo] * river_weight

    # Pump-pump, pump-river, river-river
    adj_matrix[num_piezo:num_piezo + num_pump, num_piezo:num_piezo + num_pump] = sub_adj[pump_sub_start:pump_sub_end, pump_sub_start:pump_sub_end] * pump_weight
    adj_matrix[num_piezo:num_piezo + num_pump, river_adj_start:] = sub_adj[pump_sub_start:pump_sub_end, river_sub_start:] * max(pump_weight, river_weight)
    adj_matrix[river_adj_start:, num_piezo:num_piezo + num_pump] = sub_adj[river_sub_start:, pump_sub_start:pump_sub_end] * max(pump_weight, river_weight)
    adj_matrix[river_adj_start:, river_adj_start:] = sub_adj[river_sub_start:, river_sub_start:] * river_weight

    # ── 4. Precipitation and evaporation connections (distance-based, fixed weights) ──
    dist_matrix = pairwise_distances(all_coords)
    for i in range(num_piezo):
        prec_idx = num_piezo + num_pump + np.argmin(
            dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
        adj_matrix[i, prec_idx] = 0.3

        evap_idx = num_piezo + num_pump + num_prec + np.argmin(
            dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
        adj_matrix[i, evap_idx] = 0.4

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    # ── 5. Print stats ──
    piezo_edges = np.count_nonzero(adj_matrix[:num_piezo, :num_piezo]) // 2
    edges_per_node = np.count_nonzero(adj_matrix[:num_piezo, :num_piezo], axis=1)
    total_edges = np.count_nonzero(adj_matrix) // 2
    print(f"  Feature-distance graph (radius={radius}):")
    print(f"    Piezo-piezo edges: {piezo_edges}")
    print(f"    Edges/piezo: min={edges_per_node.min()}, mean={edges_per_node.mean():.1f}, max={edges_per_node.max()}")
    print(f"    Total edges (inc. exo): {total_edges}")

    return adj_matrix


def generate_mixed_optimal_adjacency(rmse_table_path, variant_adj_matrices, num_piezo,
                                     fallback_variant='default'):
    """Build a mixed adjacency matrix by selecting each piezometer node's row
    from whichever graph variant produced the lowest seed-averaged RMSE.

    Parameters
    ----------
    rmse_table_path : str or Path
        Path to the Excel file with per-node RMSE values.
        Expected sheet: "Per-Node RMSE Comparison", header at row 3.
        Columns: Variant, Seed, F_w, Overall RMSE, then one column per node.
    variant_adj_matrices : dict[str, np.ndarray]
        Mapping from variant name (as it appears in the RMSE table) to its
        adjacency matrix (numpy array).
    num_piezo : int
        Number of piezometer nodes (first num_piezo rows/cols in adj matrix).
    fallback_variant : str
        Variant to use for exogenous node rows and any node not found in table.

    Returns
    -------
    mixed_adj : np.ndarray
        Symmetric mixed adjacency matrix.
    best_variant_per_node : dict
        Mapping from node name to its best variant.
    """
    # Read RMSE table
    df = pd.read_excel(rmse_table_path, sheet_name="Per-Node RMSE Comparison",
                       header=2)

    # Node columns are everything after the first 4 metadata columns
    node_cols = df.columns[4:].tolist()

    # Compute mean RMSE per variant per node (averaging across seeds)
    variant_means = df.groupby(df.columns[0])[node_cols].mean()

    # For each node, find the variant with the lowest mean RMSE
    best_variant_per_node = variant_means.idxmin(axis=0).to_dict()

    num_total = variant_adj_matrices[fallback_variant].shape[0]

    # Build mixed adjacency
    mixed_adj = np.zeros((num_total, num_total), dtype=float)

    # Map node names to indices (first num_piezo nodes are piezometers)
    for i, node_name in enumerate(node_cols[:num_piezo]):
        best_v = best_variant_per_node.get(node_name, fallback_variant)
        if best_v in variant_adj_matrices:
            mixed_adj[i, :] = variant_adj_matrices[best_v][i, :]
        else:
            mixed_adj[i, :] = variant_adj_matrices[fallback_variant][i, :]

    # Exogenous rows from fallback variant
    mixed_adj[num_piezo:, :] = variant_adj_matrices[fallback_variant][num_piezo:, :]

    # Symmetrize
    mixed_adj = np.maximum(mixed_adj, mixed_adj.T)

    return mixed_adj, best_variant_per_node


def main(df_piezo_columns, pump_columns, locations_no_missing, graph_type, percentage=None, n_piezo_connected=3, feature_importance_multiplier = None, n_pumps_connected = 4, weight_mode = 'fixed', same_layer = False, directed_graph=False, mean_gw_elevation=None, rf_weight_min=0.08, rf_weight_max=0.2, rf_vim_min=0.01, rf_min_connections=3, rmse_table_path=None, variant_graph_paths=None, sp_config=None, fd_config=None, rf_config=None, exo_ablation=None, log_pump_config=None, one_way_exo=False):
    # Paths to the metadata files (update these paths according to your folder structure)

    metadata_path = PIEZO_METADATA
    pump_metadata_path = PUMP_METADATA
    evap_metadata_path = EVAP_METADATA
    prec_metadata_path = PREC_METADATA
    river_metadata_path = RIVER_METADATA
    outdir = Path(PREPROCESSED_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    all_x, all_y, all_z, all_type, num_piezo, num_pump, num_prec, num_evap, num_river, node_names = load_and_concatenate_metadata(
        metadata_path, pump_metadata_path, evap_metadata_path, prec_metadata_path, river_metadata_path,
        df_piezo_columns, pump_columns, locations_no_missing
    )

    # Prepare static features
    static_features = create_static_features(all_x, all_y, all_z, all_type)

    # Build node type labels from index ranges
    type_map = {1: 'Piezometer', 2: 'Pump', 3: 'Precipitation', 4: 'Evaporation', 5: 'River'}
    type_labels = [type_map[int(t)] for t in all_type]

    # build a DataFrame and export
    nodes_df = pd.DataFrame({"name": node_names, "x": all_x, "y": all_y, "z": all_z, "Type": type_labels})

    # Merge geolayer/regis_layer info for piezometers if available
    if Path(PIEZO_LAYER_INFORMATION).exists():
        layer_df = pd.read_csv(PIEZO_LAYER_INFORMATION)[['name', 'geolayer', 'regis_layer']].drop_duplicates(subset='name')
        nodes_df = nodes_df.merge(layer_df, on='name', how='left')

    nodes_df.to_csv(outdir / "nodes.csv", index=False)

    print(f"Wrote {len(node_names)} names to {outdir/'node_names.csv'}")

    # Generate adjacency matrix
    # For this, you need to adjust coordinates format and threshold as needed
    coordinates = np.stack((all_x, all_y), axis=-1)
    # adj_matrix = generate_adjacency_matrix(coordinates, threshold=0.5)  # Adjust threshold as needed
    #adj_matrix = generate_complex_adjacency_matrix(coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, percentage, n_piezo_connected)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    GENERATED_GRAPHS.mkdir(parents=True, exist_ok=True)

    # Descriptive graph filename parts shared by all types
    graph_tag = (f"adj_{graph_type}_WM_{weight_mode}"
                 f"_piezo_{n_piezo_connected}_pumps_{n_pumps_connected}")
    if same_layer:
        graph_tag += "_samelayer"
    if directed_graph:
        graph_tag += "_directed"
    graph_tag += f"_{timestamp}"

    if graph_type == 'default':
        adj_matrix = generate_complex_adjacency_matrix(coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, percentage, n_piezo_connected, n_pumps_connected)
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)
    elif graph_type == 'geolayer':
        adj_matrix = generate_layer_constrained_adjacency_matrix(coordinates, df_piezo_columns, num_piezo, num_pump, num_prec, num_evap, num_river, layer_column='geolayer', percentage=percentage, n_piezo_connected=n_piezo_connected, n_pumps_connected=n_pumps_connected, weight_mode=weight_mode)
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)
    elif graph_type == 'regis_layer':
        adj_matrix = generate_layer_constrained_adjacency_matrix(coordinates, df_piezo_columns, num_piezo, num_pump, num_prec, num_evap, num_river, layer_column='regis_layer', percentage=percentage, n_piezo_connected=n_piezo_connected, n_pumps_connected=n_pumps_connected, weight_mode=weight_mode)
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)
    elif graph_type == 'rf':
          if weight_mode == 'fixed':
              adj_matrix = generate_rf_adjacency_fixed(
                  df_piezo_columns, coordinates, num_piezo, num_pump, num_prec,
                  num_evap, num_river, n_piezo_connected, n_pumps_connected,
                  layer_constrain=same_layer)
          elif weight_mode == 'variable':
              adj_matrix = generate_rf_adjacency_variable(
                  df_piezo_columns, coordinates, num_piezo, num_pump, num_prec,
                  num_evap, num_river, n_top_connections=n_piezo_connected,
                  feature_importance_multiplier=feature_importance_multiplier,
                  n_pumps_connected=n_pumps_connected,
                  rf_weight_min=rf_weight_min, rf_weight_max=rf_weight_max,
                  layer_constrain=same_layer)
          elif weight_mode == 'full':
              adj_matrix = generate_rf_full_vim_matrix(
                  df_piezo_columns, coordinates, num_piezo, num_pump, num_prec,
                  num_evap, num_river, vim_min=rf_vim_min,
                  n_top_connections=n_piezo_connected if n_piezo_connected != 3 else None,
                  n_min_connections=rf_min_connections,
                  n_pumps_connected=n_pumps_connected,
                  feature_importance_multiplier=feature_importance_multiplier,
                  rf_weight_min=rf_weight_min, rf_weight_max=rf_weight_max)
          elif weight_mode == 'cutoff':
              rc = rf_config or {}
              adj_matrix = generate_rf_cutoff_adjacency(
                  coordinates, num_piezo, num_pump, num_prec,
                  num_evap, num_river, rc)
          np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)

    elif graph_type == 'shortest_path':
        sp = sp_config or {}
        adj_matrix = generate_shortest_path_adjacency(
            coordinates, df_piezo_columns, num_piezo, num_pump, num_prec,
            num_evap, num_river,
            resistance_source=sp.get('resistance_source', 'regis'),
            n_piezo_connected=n_piezo_connected,
            sp_min_sensitivity=sp.get('sp_min_sensitivity', 0.0),
            sp_min_connections=sp.get('sp_min_connections', 3),
            piezo_weight_range=sp.get('piezo_weight_range', (rf_weight_min, rf_weight_max)),
            pump_weight_range=sp.get('pump_weight_range', (0.15, 0.25)),
            river_weight_range=sp.get('river_weight_range', (0.4, 0.6)),
            n_pumps_connected=n_pumps_connected,
            n_rivers_connected=sp.get('n_rivers_connected', 2),
            include_pumps=sp.get('include_pumps', True),
            include_rivers=sp.get('include_rivers', True),
            use_hydraulic_exo=sp.get('use_hydraulic_exo', False),
        )
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)

    elif graph_type == 'feature_distance':
        fd = fd_config or {}
        adj_matrix = generate_feature_distance_adjacency(
            coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, fd,
        )
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)

    elif graph_type == 'mixed':
        if rmse_table_path is None or variant_graph_paths is None:
            raise ValueError("graph_type='mixed' requires rmse_table_path and variant_graph_paths (dict of variant→np.ndarray)")
        adj_matrix, best_variants = generate_mixed_optimal_adjacency(
            rmse_table_path, variant_graph_paths, num_piezo)
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)
        # Log variant selection summary
        from collections import Counter
        counts = Counter(best_variants.values())
        print(f"Mixed graph: selected best variant per node from {len(variant_graph_paths)} variants")
        for v, c in counts.most_common():
            print(f"  {v}: {c} nodes")

    elif graph_type == 'prebuilt':
        # Load a pre-built adjacency matrix from a user-specified .npy file
        prebuilt_path = sp_config.get('prebuilt_path') if sp_config else None
        if prebuilt_path is None:
            raise ValueError("graph_type='prebuilt' requires sp_config={'prebuilt_path': '/path/to/adj.npy'}")
        adj_matrix = np.load(prebuilt_path)
        print(f"  Loaded prebuilt adjacency from {prebuilt_path}: shape={adj_matrix.shape}, nnz={np.count_nonzero(adj_matrix)}")
        np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # ── Custom pump weights (Thiem or coherence) ──
    if log_pump_config is not None:
        lpc = log_pump_config
        source = lpc.get('source', 'thiem')

        if source == 'per_station':
            # Per-station fixed weights: assign a specific weight per pump station.
            # lpc['station_weights'] = {'Fikkersdries': 0.3, 'Zetten': 0.1, ...}
            station_weights = lpc['station_weights']

            # Optionally apply coherence connectivity first (replaces distance-based edges)
            if lpc.get('coherence_connectivity'):
                coh_band = lpc.get('band')
                if coh_band:
                    from config import INPUT_DIR
                    coh_file = INPUT_DIR / "wells" / f"pump_weights_coherence_{coh_band}.csv"
                else:
                    coh_file = PUMP_COHERENCE_WEIGHTS
                coh_weights = compute_coherence_pump_weights(num_piezo, num_pump,
                                                             coherence_weights_file=coh_file)
                adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump] = coh_weights

            # Override weights per station.
            # connect_all=True: listed pumps connect to ALL piezometers.
            # weight=0: disconnect that pump entirely.
            connect_all = lpc.get('connect_all', False)
            pump_block = adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump]
            for j, pname in enumerate(pump_columns):
                w = station_weights.get(pname, 0.2)
                if w == 0:
                    pump_block[:, j] = 0
                elif connect_all:
                    pump_block[:, j] = w
                else:
                    mask = pump_block[:, j] > 0
                    pump_block[mask, j] = w
            adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump] = pump_block
        elif source == 'coherence':
            # Load pre-computed coherence weights (200 × 4 matrix with zeros for disconnected)
            # Optional band suffix selects a band-specific CSV (e.g. '90_365d', 'gt365d')
            band_suffix = lpc.get('band')
            if band_suffix:
                from config import INPUT_DIR
                coh_file = INPUT_DIR / "wells" / f"pump_weights_coherence_{band_suffix}.csv"
            else:
                coh_file = PUMP_COHERENCE_WEIGHTS
            coh_weights = compute_coherence_pump_weights(num_piezo, num_pump,
                                                         coherence_weights_file=coh_file)
            adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump] = coh_weights
        else:
            # Default: Thiem log-scaled weights with R cutoff
            thiem_weights = compute_log_pump_weights(
                num_piezo,
                n_pumps_connected=n_pumps_connected,
                multiplier=lpc.get('multiplier', 1.0),
                w_min=lpc.get('w_min', 0.1),
                w_max=lpc.get('w_max', 0.3),
                R=lpc.get('R', 10000),
                per_station_R=lpc.get('per_station_R'),
            )
            adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump] = thiem_weights

        # Symmetrise pump edges
        adj_matrix[num_piezo:num_piezo + num_pump, :num_piezo] = (
            adj_matrix[:num_piezo, num_piezo:num_piezo + num_pump].T
        )

    # ── Exogenous ablation: selectively remove node-type edges post-hoc ──
    if exo_ablation:
        pump_start = num_piezo
        pump_end = num_piezo + num_pump
        prec_start = pump_end
        prec_end = prec_start + num_prec
        evap_start = prec_end
        evap_end = evap_start + num_evap
        river_start = num_piezo + num_pump + num_prec + num_evap
        river_end = river_start + num_river

        if exo_ablation.get('remove_pumps'):
            adj_matrix[:, pump_start:pump_end] = 0
            adj_matrix[pump_start:pump_end, :] = 0
            print("  Exo ablation: removed all pump edges")
        if exo_ablation.get('remove_precip'):
            adj_matrix[:, prec_start:prec_end] = 0
            adj_matrix[prec_start:prec_end, :] = 0
            print("  Exo ablation: removed all precipitation edges")
        if exo_ablation.get('remove_evap'):
            adj_matrix[:, evap_start:evap_end] = 0
            adj_matrix[evap_start:evap_end, :] = 0
            print("  Exo ablation: removed all evaporation edges")
        if exo_ablation.get('remove_rivers'):
            adj_matrix[:, river_start:river_end] = 0
            adj_matrix[river_start:river_end, :] = 0
            print("  Exo ablation: removed all river edges")

        remaining = np.count_nonzero(adj_matrix)
        print(f"  Exo ablation: {remaining} non-zero entries remaining")

    # ── One-way exogenous: exo → piezo only (prevent piezo signal relaying through exo nodes) ──
    if one_way_exo:
        exo_start = num_piezo
        before_nnz = np.count_nonzero(adj_matrix)
        # Zero out exo←piezo edges (exo rows, piezo cols): prevents piezo signal flowing into exo nodes
        adj_matrix[exo_start:, :num_piezo] = 0
        # Zero out exo←exo edges: prevents exo nodes relaying through each other
        adj_matrix[exo_start:, exo_start:] = 0
        after_nnz = np.count_nonzero(adj_matrix)
        print(f"  One-way exo: removed {before_nnz - after_nnz} reverse/exo-exo edges "
              f"({before_nnz} → {after_nnz} non-zero)")

    # Apply directional mask: keep piezo-piezo edges only from higher to lower GW elevation
    if directed_graph and mean_gw_elevation is not None:
        elev_mask = mean_gw_elevation[:, None] >= mean_gw_elevation[None, :]  # (num_piezo, num_piezo)
        adj_matrix[:num_piezo, :num_piezo] *= elev_mask
        n_directed = np.count_nonzero(adj_matrix[:num_piezo, :num_piezo])
        print(f"Directed graph: {n_directed} piezo-piezo edges (higher to lower GW elevation)")

    base_data_path = PREPROCESSED_DIR

    # Save adj_matrix and static_features for later use in your GNN model
    torch.save(adj_matrix, base_data_path/ 'adj_matrix.pt')
    torch.save(static_features, base_data_path / 'static_features.pt')

    adj_matrix_tensor = torch.tensor(adj_matrix).float()
    static_features_tensor = static_features.clone().detach()

    return adj_matrix_tensor, static_features_tensor

if __name__ == "__main__":
    main()
