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
    HYDRAULIC_RESISTANCE_DIR
)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_n_closest_pumps_indices(n, num_piezo, pump_distances_file=PUMP_DISTANCES):
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
        Path to a CSV file (no index column) of shape (num_piezometers, num_pumps).

    Returns
    -------
    np.ndarray
        Array of shape (num_piezometers, n) where row i contains the adjacency-matrix
        indices of the n closest pumps to piezometer i.
    """
    # load raw distances; shape = (num_piezometers, num_pumps)
    dist_array = pd.read_csv(pump_distances_file, header=None).values

    # argsort each row to get pump indices in ascending distance order
    sorted_pump_indices = np.argsort(dist_array, axis=1)

    # take the first n indices for each row, offset by num_piezo
    return sorted_pump_indices[:, :n] + num_piezo


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
        n_pumps_connected: int = 4,
        feature_importance_multiplier: float = 1.0,
        rf_weight_min: float = 0.08,
        rf_weight_max: float = 0.2
) -> np.ndarray:
    """
    Full RF adjacency: keep ALL piezo-piezo connections where
    RF importance >= vim_min. Optionally also limit to top-N per node.
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

    print(f"VIM threshold: {vim_min}, top-N cap: {n_top_connections}")

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


def generate_hydraulic_adjacency_matrix(
    all_coords, piezo_names, num_piezo, num_pump, num_prec, num_evap, num_river,
    n_piezo_connected=3, n_pumps_connected=4, resistance_mode='dijkstra',
    weight_min=0.08, weight_max=0.2
):
    """
    Build adjacency matrix using pre-computed hydraulic resistance from GeoTOP.

    The piezo-piezo block uses the resistance matrix computed in the
    hydraulic_connectivity_graph notebook.  Each piezometer connects to its
    top-N lowest-resistance (= highest connectivity) neighbors, with edge
    weights scaled to [weight_min, weight_max].  Exogenous connections
    (pumps, precip, evap, rivers) follow the same distance-based rules as
    the default graph.

    Parameters
    ----------
    resistance_mode : str
        'dijkstra' or 'straight' — selects which resistance .npy file to load.
    """
    # ── Load pre-computed resistance matrix (piezo-only, ordered by metadata CSV) ──
    res_file = HYDRAULIC_RESISTANCE_DIR / f'resistance_{resistance_mode}.npy'
    res_full = np.load(res_file)

    # The notebook built the resistance matrix using piezometer_metadata.csv order.
    # Re-order rows/cols to match piezo_names (= df_piezo_columns order).
    meta = pd.read_csv(PIEZO_METADATA)
    notebook_order = meta['name'].tolist()

    # Build index mapping: notebook_order position → piezo_names position
    nb_name_to_idx = {n: i for i, n in enumerate(notebook_order)}
    reorder = [nb_name_to_idx[n] for n in piezo_names]
    res = res_full[np.ix_(reorder, reorder)]

    # ── Convert resistance → top-N adjacency ──
    connectivity = np.where(res > 0, 1.0 / res, 0.0)
    np.fill_diagonal(connectivity, 0)

    piezo_block = np.zeros((num_piezo, num_piezo))
    for i in range(num_piezo):
        row = connectivity[i]
        top_idx = np.argsort(row)[-n_piezo_connected:]
        top_idx = top_idx[row[top_idx] > 0]  # exclude zeros
        piezo_block[i, top_idx] = row[top_idx]

    # Symmetrize
    piezo_block = np.maximum(piezo_block, piezo_block.T)

    # Scale non-zero weights to [weight_min, weight_max]
    pos = piezo_block > 0
    if pos.any():
        vals = piezo_block[pos]
        scaled = weight_min + (vals - vals.min()) / (vals.max() - vals.min() + 1e-10) * (weight_max - weight_min)
        piezo_block[pos] = scaled

    # ── Build full adjacency with exogenous connections ──
    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[:num_piezo, :num_piezo] = piezo_block

    dist_matrix = pairwise_distances(all_coords)
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, num_piezo)

    for i in range(num_piezo):
        # Pumps
        adj_matrix[i, closest_pumps[i]] = 0.2

        # Closest precipitation
        prec_idx = num_piezo + num_pump + np.argmin(
            dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
        adj_matrix[i, prec_idx] = 0.3

        # Closest evaporation
        evap_idx = num_piezo + num_pump + num_prec + np.argmin(
            dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
        adj_matrix[i, evap_idx] = 0.4

        # Two closest rivers
        river_indices = np.argsort(dist_matrix[i, -num_river:])[:2] + (num_nodes - num_river)
        adj_matrix[i, river_indices] = 0.5

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
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


def main(df_piezo_columns, pump_columns, locations_no_missing, graph_type, percentage=None, n_piezo_connected=3, feature_importance_multiplier = None, n_pumps_connected = 4, weight_mode = 'fixed', same_layer = False, directed_graph=False, mean_gw_elevation=None, rf_weight_min=0.08, rf_weight_max=0.2, rf_vim_min=0.01, rmse_table_path=None, variant_graph_paths=None):
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
                  n_pumps_connected=n_pumps_connected,
                  feature_importance_multiplier=feature_importance_multiplier,
                  rf_weight_min=rf_weight_min, rf_weight_max=rf_weight_max)
          np.save(GENERATED_GRAPHS / f"{graph_tag}.npy", adj_matrix)

    elif graph_type in ('hydraulic_dijkstra', 'hydraulic_straight'):
        mode = 'dijkstra' if graph_type == 'hydraulic_dijkstra' else 'straight'
        adj_matrix = generate_hydraulic_adjacency_matrix(
            coordinates, df_piezo_columns, num_piezo, num_pump, num_prec,
            num_evap, num_river, n_piezo_connected=n_piezo_connected,
            n_pumps_connected=n_pumps_connected,
            resistance_mode=mode,
            weight_min=rf_weight_min, weight_max=rf_weight_max)
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

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

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
