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
    RF_TRAINED_ALL, RF_TRAINED_PIEZOS_ONLY, PIEZO_LAYER_INFORMATION
)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_n_closest_pumps_indices(n, pump_distances_file=PUMP_DISTANCES):
    """
    Reads a CSV of piezometer‐to‐pump distances and returns, for each piezometer,
    the integer column indices of the n closest pumps.

    Parameters
    ----------
    n : int
        Number of closest pumps to find per piezometer.
    pump_distances_file : str
        Path to a CSV file (no index column) of shape (num_piezometers, num_pumps).

    Returns
    -------
    np.ndarray
        Array of shape (num_piezometers, n) where row i contains the pump‐column
        indices (0-based) of the n closest pumps to piezometer i.
    """
    # load raw distances; shape = (num_piezometers, num_pumps)
    dist_array = pd.read_csv(pump_distances_file, header=None).values

    # argsort each row to get pump indices in ascending distance order
    sorted_pump_indices = np.argsort(dist_array, axis=1)

    # take the first n indices for each row
    return sorted_pump_indices[:, :n] + 199 #add 200 for piezo offset.  


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


def penalize_exogenous_weights(
    adj_matrix: np.ndarray,
    piezo_names: list,
    layer_csv: str,
    layer_column: str,
    target_layer: str,
    exo_penalties: dict,
) -> np.ndarray:
    """
    Apply penalty percentages to multiple types of exogenous connections
    for piezometers in a specified geolayer.
    """
    df = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index("name")
    labels = df.loc[piezo_names, "geolayer"].fillna("MISSING").values

    type_ranges = {
        'pump':  list(range(200, 204)),
        'prec':  list(range(204, 206)),
        'evap':  list(range(206, 208)),
        'river': list(range(208, 212))
    }
    for exo_type in exo_penalties:
        if exo_type not in type_ranges:
            raise ValueError(f"Invalid exo_type '{exo_type}' in penalties dict")

    for i, layer in enumerate(labels):
        if layer != target_layer:
            continue
        for exo_type, penalty in exo_penalties.items():
            for j in type_ranges[exo_type]:
                adj_matrix[i, j] *= (1 - penalty)
                adj_matrix[j, i] *= (1 - penalty)

    return adj_matrix


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
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, pump_distances_file=PUMP_DISTANCES) 

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
    adj_matrix = adj_matrix + adj_matrix.T

    return adj_matrix


import pandas as pd
import numpy as np


def generate_layer_constrained_adjacency_matrix(
    all_coords, piezo_names, num_piezo, num_pump, num_prec, num_evap, num_river,
    layer_column="geolayer", percentage=None, n_piezo_connected=3, n_pumps_connected = 4, exo_penalties = False, rf_perturb = True
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
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, pump_distances_file=PUMP_DISTANCES) 

    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))


    aq2_indices_list = []
    if exo_penalties:
      for i in range(len(piezo_names)):
        if layer_df.loc[piezo_names[i], "geolayer"] == "aq2": aq2_indices_list.append(i)  #Now you have a list of indices of piezometer indices for aq2.


    if rf_perturb:
      rf_feature_importance = joblib.load(RF_TRAINED_PIEZOS_ONLY)
      P = rf_feature_importance.shape[0]
      if P != num_piezo:
        raise ValueError(
            f"Expected RF importances shape ({num_piezo},{num_piezo}), got {rf_feature_importance.shape}"
        )

      mask = rf_feature_importance> 0
      nonzero = rf_feature_importance[mask]
      min_w, max_w = nonzero.min(), nonzero.max()

      rf_scaled = np.zeros_like(rf_feature_importance)
      rf_scaled[mask] = (
          (rf_feature_importance[mask] - min_w)
          / (max_w - min_w)                 # now in [0,1]
          * (0.16 - 0.08)                    # now in [0,0.1]
          + 0.08                            # now in [0.08,0.12]
      )

      # 3. use rf_scaled instead of rf_full
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
            if rf_perturb:
              adj_matrix[i, nearest] = rf_full[i, nearest]
            else: adj_matrix[i, nearest] = 0.1
              


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
    adj_matrix = adj_matrix + adj_matrix.T

    return adj_matrix
"""

def generate_layer_constrained_adjacency_matrix(
        all_coords, piezo_names, num_piezo, num_pump, num_prec, num_evap, num_river,
        layer_column="geolayer", percentage=None, n_piezo_connected=3, n_pumps_connected=4
):
    
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

    from preprocessing import euclidean_distance

    # Load geolayer/regis_layer info
    layer_df = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index("name")
    layer_labels = layer_df.loc[piezo_names, layer_column].fillna("MISSING").values

    # Read pump distances
    pumps = pd.read_csv(PUMP_DISTANCES, header=0, index_col=0)
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, pump_distances_file=PUMP_DISTANCES)

    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))

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
            adj_matrix[i, prec_idx] = 0.3
            adj_matrix[i, evap_idx] = 0.4
            adj_matrix[i, river_idxs] = 0.5

    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix  # Symmetric adjacency matrix
"""

def generate_fixed_layer_constrained_rf_adjacency_matrix_layer(
    piezo_columns: list,
    all_coords: np.ndarray,
    num_piezo: int,
    num_pump: int,
    num_prec: int,
    num_evap: int,
    num_river: int,
    n_top_connections: int = 3,
    n_pumps_connected: int = 4,
    ignore_layers = True
) -> np.ndarray:
    """
    #NOTE THIS DOESN'T ESTABLISH EXO WEIGHT CONNECTIONS THROUGH RF, POSSIBLY TRY THAT
    Generate adjacency matrix using RF feature importance between piezometers only.
    Piezo→piezo edges: top-n same-layer peers (fixed weight 0.1).
    Optionally adds edges to exogenous nodes based on spatial proximity:
      - pumps (0.2), precipitation (0.3), evaporation (0.4), rivers (0.5).
    """
    num_nodes = all_coords.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Load layer labels for piezometers
    df_layers = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index('name')
    piezo_labels = df_layers.loc[piezo_columns, "geolayer"].fillna('MISSING').values

    # Load RF importance matrix (P×P)
    rf_feature_importance = joblib.load(RF_TRAINED_PIEZOS_ONLY)
    P = rf_feature_importance.shape[0]
    if P != num_piezo:
        raise ValueError(
            f"Expected RF importances shape ({num_piezo},{num_piezo}), got {rf_feature_importance.shape}"
        )

    # Read closest pumps for exogenous
    closest_pumps = get_n_closest_pumps_indices(
        n_pumps_connected,
        pump_distances_file=PUMP_DISTANCES
    )

    # 1) Piezo→piezo: top-n importances among same-layer peers
    for i in range(num_piezo):
        peers = [j for j in range(num_piezo)
                 if j != i and piezo_labels[j] == piezo_labels[i]]
        if not peers:
            continue
        weights = rf_feature_importance[i, peers]
        topk_rel = np.argsort(weights)[-n_top_connections:][::-1]
        selected = [peers[idx] for idx in topk_rel]
        for j in selected:
            adj_matrix[i, j] = 0.1

    # 2) Exogenous connections (pumps, etc.)
    for i in range(num_piezo):
        # pumps
        pump_idxs = closest_pumps[i].tolist()
        adj_matrix[i, pump_idxs] = 0.2

        # distances to all nodes
        dist = np.linalg.norm(all_coords[i] - all_coords, axis=1)

        #if not ignore_layers
        # precipitation
        if num_prec > 0:
            prec_slice = dist[num_piezo + num_pump:num_piezo + num_pump + num_prec]
            prec_idx = num_piezo + num_pump + np.argmin(prec_slice)
            adj_matrix[i, prec_idx] = 0.3

        # evaporation
        if num_evap > 0:
            evap_slice = dist[num_piezo + num_pump + num_prec:
                               num_piezo + num_pump + num_prec + num_evap]
            evap_idx = num_piezo + num_pump + num_prec + np.argmin(evap_slice)
            adj_matrix[i, evap_idx] = 0.4

        # rivers (two closest)
        if num_river > 0:
            river_slice = dist[-num_river:]
            two = np.argsort(river_slice)[:2] + (num_nodes - num_river)
            adj_matrix[i, two] = 0.5

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix # Symmetric adjacency matrix

def generate_rf_adjacency_matrix(
        piezo_columns: list,
        all_coords: np.ndarray,
        num_piezo: int,
        num_pump: int,
        num_prec: int,
        num_evap: int,
        num_river: int,
        n_top_connections: int = 3,
        n_pumps_connected: int = 4
) -> np.ndarray:
    """
    Generate adjacency matrix using Random Forest feature importance between piezometers only.
    Optionally adds edges to exogenous nodes based on spatial proximity.

    Parameters:
    - time_series_df: DataFrame with time series for all piezometers (columns = piezo IDs)
    - piezo_columns: list of piezometer column names
    - all_coords: np.array of shape (N, 2) with X, Y for all nodes
    - num_piezo, num_pump, ..., num_river: counts of each node type
    - n_top_connections: number of strongest connections to retain per node
    - percentage: optionally limit connections to % of total nodes

    Returns:
    - Adjacency matrix (numpy array)
    """
    num_nodes = all_coords.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Read pump distances
    pumps = pd.read_csv(PUMP_DISTANCES, header=0, index_col=0)
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, pump_distances_file=PUMP_DISTANCES)

    # 1) Load your saved P×P importance matrix
    rf_feature_importance = joblib.load(RF_TRAINED_PIEZOS_ONLY)  # shape (P, P)
    P = rf_feature_importance.shape[0]
    if P != len(piezo_columns):
        raise ValueError(
            f"Expected RF importances of shape ({len(piezo_columns)},{len(piezo_columns)}), "
            f"got {rf_feature_importance.shape}"
        )

    # 2) For each piezo i, keep its top-n importances to other piezos
    for i in range(P):
        row = rf_feature_importance[i]
        # indices of the n largest values in row
        top_idxs = np.argsort(row)[-n_top_connections:][::-1]
        # assign scaled importances
        for j in top_idxs:
            adj_matrix[i, j] = 0.1

    # Add connections to exogenous nodes (pumps, etc.)

    selected_indices = range(num_nodes)

    for i in selected_indices:
        if i < num_piezo:
            dist_matrix = np.linalg.norm(all_coords[i] - all_coords, axis=1)
            pump_indices = closest_pumps[i].tolist()
            prec_index = num_piezo + num_pump + np.argmin(
                dist_matrix[num_piezo + num_pump: num_piezo + num_pump + num_prec])
            evap_index = num_piezo + num_pump + num_prec + np.argmin(
                dist_matrix[num_piezo + num_pump + num_prec: num_piezo + num_pump + num_prec + num_evap])
            river_indices = np.argsort(dist_matrix[-num_river:])[:2] + (num_nodes - num_river)

            adj_matrix[i, pump_indices] = 0.2
            adj_matrix[i, prec_index] = 0.3
            adj_matrix[i, evap_index] = 0.4
            adj_matrix[i, river_indices] = 0.5

    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix # Symmetric adjacency matrix

def generate_rf_adjacency_variable_weights_matrix_layer_constrained(
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
        multiply_exo_weights = True,
        rf_piezo_scale = True #This makes it more similar to a geolayer with slight perterbations, described below.
) -> np.ndarray:
    """
    Generate adjacency matrix using Random Forest feature importance between all nodes.
    Piezo→piezo edges only connect to the top-n same-layer peers. Exogenous edges use RF weights.
    """
    # load the FULL RF matrix (must be N×N)
    rf_full = joblib.load(RF_TRAINED_ALL)  # shape (N, N)
    N = rf_full.shape[0]
    if rf_full.shape != (N, N):
        raise ValueError(f"Expected full RF matrix of shape ({N},{N}), got {rf_full.shape}")


    # scale once
    #rf_full = rf_full * feature_importance_multiplier
    nonzero = rf_full[rf_full > 0]
    if nonzero.size == 0:
      raise ValueError("No nonzero importances found in rf_full.")

    min_w, max_w = nonzero.min(), nonzero.max()

    # 2. build a scaled copy (zeros stay zero)
    if rf_piezo_scale:
      rf_scaled = np.zeros_like(rf_full)
      rf_scaled[rf_full > 0] = (
          (rf_full[rf_full > 0] - min_w)
          / (max_w - min_w)                 # now in [0,1]
          * (0.2 - 0.08)                    # now in [0,0.12]
          + 0.08                            # now in [0.08,0.2]
      )

      # 3. use rf_scaled instead of rf_full
      rf_full = rf_scaled


    # load layer labels for piezo nodes
    df_layers = pd.read_csv(PIEZO_LAYER_INFORMATION).set_index('name')
    piezo_labels = df_layers.loc[piezo_columns, "geolayer"].fillna('MISSING').values

    # prepare adjacency and distance matrix
    adj_matrix = np.zeros((N, N), dtype=float)
    dmat = pairwise_distances(all_coords)

    # read closest pumps
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, pump_distances_file=PUMP_DISTANCES)

    # piezo→piezo: top-k within same layer
    for i in range(num_piezo):
        # find same-layer peers
        peers = [j for j in range(num_piezo) if j != i and piezo_labels[j] == piezo_labels[i]]
        if not peers:
            continue
        weights = rf_full[i, peers]
        # pick the top-n weights
        topk_idx = np.argsort(weights)[-n_top_connections:][::-1]
        selected = [peers[k] for k in topk_idx]
        adj_matrix[i, selected] = weights[topk_idx] * feature_importance_multiplier  #IS THIS ACTUALLY ASSIGNING PROPER WEIGHTS?

    # compute global start indices
    start_pump = num_piezo
    start_prec = start_pump + num_pump
    start_evap = start_prec + num_prec
    start_river = start_evap + num_evap

    for i in range(num_piezo):
    # 1) pumps
      pump_idxs = np.array(closest_pumps[i], dtype=int)
      #print(f"i={i}: pump_idxs = {pump_idxs}")
      #if len(pump_idxs) == 0:
      #    print(f"  >> WARNING: no pumps connected to piezo {i}")
      adj_matrix[i, pump_idxs] = 0.2 #Set multiply_ex0_weights to false for original exo weights to remain
      if multiply_exo_weights: adj_matrix[i, pump_idxs] *= 10

      # 2) compute distances
      dist = np.linalg.norm(all_coords[i] - all_coords, axis=1)
      #print(f"i={i}: dist.shape = {dist.shape}")

      # 3) precipitation
      if num_prec > 0:
          start = num_piezo + num_pump
          prec_slice = dist[start : start + num_prec]
          #print(f"  prec_slice ({start}:{start+num_prec}) =", prec_slice)
          prec_rel_idx = np.argmin(prec_slice)
          prec_idx = start + prec_rel_idx
          #print(f"  prec_idx = {prec_idx}")
          adj_matrix[i, prec_idx] = 0.3
          if multiply_exo_weights: adj_matrix[i, prec_idx] *= 10 #MULT HERE ALREADY

      # 4) evaporation
      if num_evap > 0:
          start = num_piezo + num_pump + num_prec
          evap_slice = dist[start : start + num_evap]
          #print(f"  evap_slice ({start}:{start+num_evap}) =", evap_slice)
          evap_rel_idx = np.argmin(evap_slice)
          evap_idx = start + evap_rel_idx
          #print(f"  evap_idx = {evap_idx}")
          adj_matrix[i, evap_idx] = 0.4
          if multiply_exo_weights: adj_matrix[i, evap_idx] *= 10

      if num_river > 0:
          # take the num_river entries starting at start_river
          river_slice = dist[start_river : start_river + num_river]
          # pick the two closest (smallest distance)
          two_rel = np.argsort(river_slice)[:2]
          # convert back to absolute indices
          river_idxs = start_river + two_rel
          adj_matrix[i, river_idxs] = 0.5
          if multiply_exo_weights:
              adj_matrix[i, river_idxs] *= 10

    # symmetrize
    #adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    return adj_matrix
                    

    # finally symmetrize and return
    return np.maximum(adj, adj.T)

def generate_rf_adjacency_variable_weights_matrix(
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
        multiply_exo_weights = True
) -> np.ndarray:
    """
    Generate adjacency matrix using Random Forest feature importance between all nodes.
    Piezo→piezo and piezo→exo edges all pull weights from the RF matrix.
    """
    # — load the FULL RF matrix (must be N×N) —
    raw_rf = joblib.load(RF_TRAINED_ALL)
    if isinstance(raw_rf, pd.DataFrame):
        rf_full = raw_rf.values.astype(float)
    elif isinstance(raw_rf, np.ndarray):
        rf_full = raw_rf.astype(float)
    else:
        rf_full = np.array(raw_rf, dtype=float)

    N = rf_full.shape[0]
    if rf_full.shape != (N, N):
        raise ValueError(f"Expected full RF matrix shape ({N},{N}), got {rf_full.shape}")

    # scale once
    #rf_full *= feature_importance_multiplier
    print("RF matrix shape:", rf_full.shape)  
    print("Piezo count:", num_piezo, "Pump:", num_pump, 
      "Prec:", num_prec, "Evap:", num_evap, "River:", num_river)

    # prep
    adj = np.zeros((N, N), dtype=float)
    dmat = pairwise_distances(all_coords)  # (N,N) Euclidean

    # Read pump distances
    pumps = pd.read_csv(PUMP_DISTANCES, header=0, index_col=0)
    closest_pumps = get_n_closest_pumps_indices(n_pumps_connected, pump_distances_file=PUMP_DISTANCES)

    # piezo→piezo top‐k
    for i in range(num_piezo):
        row = rf_full[i, :num_piezo]  # only piezo→piezo block
        topk = np.argsort(row)[-n_top_connections:][::-1]
        adj[i, topk] = row[topk]

    # pre‐compute the global start indices
    start_pump = num_piezo
    start_prec = start_pump + num_pump
    start_evap = start_prec + num_prec
    start_river = start_evap + num_evap

    print("Start indices → pump:", start_pump,
      "prec:", start_prec,
      "evap:", start_evap,
      "river:", start_river)

    print("RF weights for piezo 0 → pumps:",
      rf_full[0, start_pump:start_pump+num_pump])
    print("RF weights for piezo 0 → precip:",
      rf_full[0, start_prec:start_prec+num_prec])
    print("RF weights for piezo 0 → evap:",
      rf_full[0, start_evap:start_evap+num_evap])
    print("RF weights for piezo 0 → rivers:",
      rf_full[0, start_river:start_river+num_river])

    for i in range(num_piezo):
        # — n closest pumps, RF weights —
        pump_idxs = np.array(closest_pumps[i], dtype=int)
        adj[i, pump_idxs] = np.maximum(
            adj[i, pump_idxs],
            rf_full[i, pump_idxs]
        )
        if multiply_exo_weights: adj[i, pump_idxs] *=2 * 10^4
        else: adj[i, pump_idxs] = 0.2 #YOLO seriously probably just scale originally and then mult. 

        # — closest precip, RF weight —
        prec_idx = start_prec + np.argmin(
            dmat[i, start_prec: start_prec + num_prec]
        )
        adj[i, prec_idx] = max(adj[i, prec_idx],
                                rf_full[i, prec_idx])
        if multiply_exo_weights: adj[i, evap_idx] *= 3 * 10^4
        else: adj[i, prec_idx] = 0.3

        # — closest evap, RF weight —
        evap_idx = start_evap + np.argmin(
            dmat[i, start_evap: start_evap + num_evap]
        )
        adj[i, evap_idx] = max(adj[i, evap_idx],
                                rf_full[i, evap_idx])
        if multiply_exo_weights: adj[i, evap_idx] *= 4 * 10^4
        else: adj[i, evap_idx] = 0.4

        # — two closest rivers, RF weights —
        riv_slice = dmat[i, start_river: start_river + num_river]
        two_rivs = np.argsort(riv_slice)[:2] + start_river
        for r in two_rivs:
            adj[i, r] = max(adj[i, r],
                            rf_full[i, r])
            if multiply_exo_weights: adj[i, r] *= 5 * 10^4
            else: adj[i, r] = 0.5

    # finally symmetrize by taking the max of (i,j) and (j,i)
    return np.maximum(adj, adj.T)


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


def main(df_piezo_columns, pump_columns, locations_no_missing, graph_type, percentage=None, n_piezo_connected=3, feature_importance_multiplier = None, n_pumps_connected = 4, weight_mode = 'fixed', same_layer = False, same_layer_kwargs = None, ext_data = True, multiply_exo_weights = False):
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

    # build a DataFrame and export
    pd.DataFrame({"name": node_names, "x":all_x, "y": all_y, "z":all_z }) \
      .to_csv(outdir / "nodes.csv", index=False)

    print(f"Wrote {len(node_names)} names to {outdir/'node_names.csv'}")

    # Generate adjacency matrix
    # For this, you need to adjust coordinates format and threshold as needed
    coordinates = np.stack((all_x, all_y), axis=-1)
    # adj_matrix = generate_adjacency_matrix(coordinates, threshold=0.5)  # Adjust threshold as needed
    #adj_matrix = generate_complex_adjacency_matrix(coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, percentage, n_piezo_connected)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if graph_type == 'default':
        adj_matrix = generate_complex_adjacency_matrix(coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, percentage, n_piezo_connected, n_pumps_connected)
        np.save(GENERATED_GRAPHS / f"adj_default_{percentage}_percent_{timestamp}.npy", adj_matrix)
    elif graph_type == 'geolayer':
        adj_matrix = generate_layer_constrained_adjacency_matrix(coordinates, df_piezo_columns, num_piezo, num_pump, num_prec, num_evap, num_river, layer_column='geolayer', percentage=percentage, n_piezo_connected = n_piezo_connected, n_pumps_connected = n_pumps_connected)
        np.save(GENERATED_GRAPHS / f"adj_geolayer_{percentage}_percent_{timestamp}.npy", adj_matrix)
    elif graph_type == 'regis_layer':
        adj_matrix = generate_layer_constrained_adjacency_matrix(coordinates, df_piezo_columns, num_piezo, num_pump, num_prec, num_evap, num_river, layer_column='geolayer', percentage=percentage, n_piezo_connected = n_piezo_connected, n_pumps_connected = n_pumps_connected)
        np.save(GENERATED_GRAPHS / f"adj_regis_{percentage}_percent_{timestamp}.npy", adj_matrix)
    elif graph_type == 'rf':
          if weight_mode == 'fixed':
            if same_layer:
              adj_matrix = generate_fixed_layer_constrained_rf_adjacency_matrix_layer(df_piezo_columns, coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, n_piezo_connected, n_pumps_connected)
              np.save(GENERATED_GRAPHS / f"adj_rf_fixed_layer_constrained_npumps:{n_pumps_connected}_n_piezo:{n_piezo_connected}_{timestamp}.npy", adj_matrix)
            else:
              adj_matrix = generate_rf_adjacency_matrix(df_piezo_columns, coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, n_piezo_connected, n_pumps_connected)
              np.save(GENERATED_GRAPHS / f"adj_rf_{n_pumps_connected}_n_piezo:{n_piezo_connected}_{timestamp}.npy", adj_matrix)
          if weight_mode == 'variable':
            if same_layer:
              adj_matrix =  generate_rf_adjacency_variable_weights_matrix_layer_constrained(df_piezo_columns, coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, n_top_connections= n_piezo_connected, n_pumps_connected = n_pumps_connected,feature_importance_multiplier = feature_importance_multiplier, multiply_exo_weights = multiply_exo_weights)
              np.save(GENERATED_GRAPHS / f"adj_rf_var_layer_constrained_{n_pumps_connected}_n_piezo:{n_piezo_connected}_{timestamp}.npy", adj_matrix)
            else: 
              adj_matrix =  generate_rf_adjacency_variable_weights_matrix(df_piezo_columns, coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, n_top_connections= n_piezo_connected, n_pumps_connected = n_pumps_connected,feature_importance_multiplier = feature_importance_multiplier, multiply_exo_weights = multiply_exo_weights)
              np.save(GENERATED_GRAPHS / f"adj_rf_var_weights_{n_pumps_connected}_n_piezo:{n_piezo_connected}_{timestamp}", adj_matrix)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


    base_data_path = PREPROCESSED_DIR

    # Save adj_matrix and static_features for later use in your GNN model
    torch.save(adj_matrix, base_data_path/ 'adj_matrix.pt')
    torch.save(static_features, base_data_path / 'static_features.pt')

    adj_matrix_tensor = torch.tensor(adj_matrix).float()
    static_features_tensor = static_features.clone().detach()

    return adj_matrix_tensor, static_features_tensor

if __name__ == "__main__":
    main()
