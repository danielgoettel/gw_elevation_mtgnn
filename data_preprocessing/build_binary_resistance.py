"""
Build a binary (aquitard/aquifer) hydraulic resistance matrix from REGIS II.

Instead of using per-cell Kh/Kv values, this assigns fixed conductivity:
  - Aquifer layers: K = 10 m/d  (low resistance conduit)
  - Aquitard layers: K = 0.001 m/d  (high resistance barrier)

The dominant control on hydraulic connectivity is whether an aquitard
sits between two nodes, not the exact K value. This simplification
captures structural connectivity.

Outputs:
  resistance_binary.npy  — 209x209 pairwise resistance matrix
  resistance_regis.npy   — 209x209 full-K resistance matrix (optional)

Usage:
    python -m data_preprocessing.build_binary_resistance [--full]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# Add parent to path so config imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    INPUT_DIR, PREPROCESSED_DIR, HYDRAULIC_RESISTANCE_DIR,
    PIEZO_METADATA, PUMP_METADATA, RIVER_METADATA
)

# Fixed K values for binary classification
K_AQUIFER = 10.0      # m/d — typical sandy aquifer
K_AQUITARD = 0.001    # m/d — typical clay aquitard

# REGIS II raster directory
REGIS_DIR = Path(__file__).resolve().parent.parent / "data" / "REGIS II Study Area Raster Data"
if not REGIS_DIR.exists():
    # Try Drive path
    REGIS_DIR = Path(r"G:\My Drive\Environmental_DL_Project\GroundwaterFlowGNN-v2-3hourly\data\REGIS II Study Area Raster Data")


def read_regis_raster(fpath):
    """Read a REGIS II GeoTIFF, masking nodata."""
    with rasterio.open(fpath) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
    mask = np.ones_like(data, dtype=bool)
    if nodata is not None:
        mask &= (data != nodata)
    mask &= (data > -9000)
    return np.where(mask, data, np.nan), transform


def discover_layers(regis_dir):
    """Discover REGIS II layers and classify as aquifer/aquitard."""
    tif_files = sorted(f for f in os.listdir(regis_dir) if f.endswith('.tif'))
    layer_files = {}
    for f in tif_files:
        base = f.replace('.tif', '')
        if base == 'MV':
            continue
        for suffix, prop in [('-t-c', 'top'), ('-b-c', 'bottom'), ('-kh-s', 'kh'), ('-kv-s', 'kv')]:
            if base.endswith(suffix):
                layer = base[:-len(suffix)]
                layer_files.setdefault(layer, {})[prop] = regis_dir / f
                break
    drawable = {k: v for k, v in layer_files.items() if 'top' in v and 'bottom' in v}
    return drawable


def load_layers(drawable):
    """Load all layer rasters and sort by depth (shallowest first)."""
    aquifer_names = sorted([k for k, v in drawable.items() if 'kh' in v])
    aquitard_names = sorted([k for k, v in drawable.items() if 'kv' in v and 'kh' not in v])
    all_names = aquifer_names + aquitard_names
    n_layers = len(all_names)

    # Read first raster to get grid geometry
    sample_data, raster_tf = read_regis_raster(list(drawable.values())[0]['top'])
    nrows, ncols = sample_data.shape
    DX = abs(raster_tf.a)

    tops = np.full((n_layers, nrows, ncols), np.nan)
    bots = np.full((n_layers, nrows, ncols), np.nan)
    K_real = np.full((n_layers, nrows, ncols), np.nan)
    is_aquifer = np.zeros(n_layers, dtype=bool)

    for li, lname in enumerate(all_names):
        files = drawable[lname]
        tops[li], _ = read_regis_raster(files['top'])
        bots[li], _ = read_regis_raster(files['bottom'])
        if 'kh' in files:
            K_real[li], _ = read_regis_raster(files['kh'])
            is_aquifer[li] = True
        elif 'kv' in files:
            K_real[li], _ = read_regis_raster(files['kv'])
            is_aquifer[li] = False

    thickness = tops - bots
    thickness = np.where(thickness > 0, thickness, np.nan)

    # Sort by median top elevation (shallowest first)
    median_tops = np.nanmedian(tops, axis=(1, 2))
    sort_order = np.argsort(-median_tops)
    all_names = [all_names[i] for i in sort_order]
    tops = tops[sort_order]
    bots = bots[sort_order]
    K_real = K_real[sort_order]
    is_aquifer = is_aquifer[sort_order]
    thickness = thickness[sort_order]

    return all_names, tops, bots, K_real, is_aquifer, thickness, raster_tf, nrows, ncols, DX


def build_graph(tops, bots, thickness, K_vals, is_aquifer, nrows, ncols, DX):
    """Build 3D hydrogeological graph and run Dijkstra.

    Parameters
    ----------
    K_vals : ndarray (n_layers, nrows, ncols)
        Conductivity values per cell. For binary mode these are fixed constants.
    """
    n_layers = len(is_aquifer)
    valid_mask = np.isfinite(thickness) & (thickness > 0) & np.isfinite(K_vals) & (K_vals > 0)
    n_valid = valid_mask.sum()

    node_id_grid = np.full((n_layers, nrows, ncols), -1, dtype=np.int32)
    node_id_grid[valid_mask] = np.arange(n_valid)

    print(f"  Valid nodes: {n_valid:,}")
    print(f"    Aquifer: {valid_mask[is_aquifer].sum():,}, Aquitard: {valid_mask[~is_aquifer].sum():,}")

    # Build edges
    edge_from, edge_to, edge_cost = [], [], []
    valid_flat = valid_mask.reshape(n_layers, -1)
    n_cells = nrows * ncols

    # Vertical edges
    print("  Building vertical edges...")
    for col_idx in range(n_cells):
        present = np.where(valid_flat[:, col_idx])[0]
        if len(present) < 2:
            continue
        r = col_idx // ncols
        c = col_idx % ncols
        for p in range(len(present) - 1):
            li, lj = present[p], present[p + 1]
            ni = node_id_grid[li, r, c]
            nj = node_id_grid[lj, r, c]
            k_i = K_vals[li, r, c]
            k_j = K_vals[lj, r, c]
            t_i = thickness[li, r, c]
            t_j = thickness[lj, r, c]
            dz = t_i / 2 + t_j / 2
            k_harm = 2.0 / (1.0 / k_i + 1.0 / k_j)
            cost = dz / k_harm
            edge_from.append(ni)
            edge_to.append(nj)
            edge_cost.append(cost)

    n_vert = len(edge_from)
    print(f"  Vertical edges: {n_vert:,}")

    # Horizontal edges (aquifer layers only)
    print("  Building horizontal edges...")
    n_horiz = 0
    for li in range(n_layers):
        if not is_aquifer[li]:
            continue
        valid_layer = valid_mask[li]
        k_layer = K_vals[li]

        # Right neighbor
        both = valid_layer[:, :-1] & valid_layer[:, 1:]
        if both.any():
            rows_h, cols_h = np.where(both)
            ni = node_id_grid[li, rows_h, cols_h]
            nj = node_id_grid[li, rows_h, cols_h + 1]
            k_harm = 2.0 / (1.0 / k_layer[rows_h, cols_h] + 1.0 / k_layer[rows_h, cols_h + 1])
            cost = DX / k_harm
            edge_from.extend(ni)
            edge_to.extend(nj)
            edge_cost.extend(cost)
            n_horiz += len(ni)

        # Down neighbor
        both = valid_layer[:-1, :] & valid_layer[1:, :]
        if both.any():
            rows_h, cols_h = np.where(both)
            ni = node_id_grid[li, rows_h, cols_h]
            nj = node_id_grid[li, rows_h + 1, cols_h]
            k_harm = 2.0 / (1.0 / k_layer[rows_h, cols_h] + 1.0 / k_layer[rows_h + 1, cols_h])
            cost = DX / k_harm
            edge_from.extend(ni)
            edge_to.extend(nj)
            edge_cost.extend(cost)
            n_horiz += len(ni)

    print(f"  Horizontal edges: {n_horiz:,}")

    # Assemble sparse symmetric graph
    all_from = np.array(edge_from, dtype=np.int32)
    all_to = np.array(edge_to, dtype=np.int32)
    all_cost = np.array(edge_cost, dtype=np.float64)
    sym_from = np.concatenate([all_from, all_to])
    sym_to = np.concatenate([all_to, all_from])
    sym_cost = np.concatenate([all_cost, all_cost])
    graph = coo_matrix((sym_cost, (sym_from, sym_to)), shape=(n_valid, n_valid)).tocsr()

    print(f"  Graph: {n_valid:,} nodes, {graph.nnz:,} directed edges")
    return graph, node_id_grid, valid_mask


def map_infrastructure_nodes(tops, bots, valid_mask, is_aquifer, all_layer_names,
                             node_id_grid, raster_tf, nrows, ncols):
    """Map piezometers, pumps, rivers to graph nodes. Returns DataFrame."""
    # Load metadata
    piezo_meta = pd.read_csv(PIEZO_METADATA)
    piezo_layers = pd.read_csv(INPUT_DIR / "piezometers" / "piezometer_layer_information.csv")
    piezo_meta = piezo_meta.merge(piezo_layers, on='name', how='left')
    piezo_meta['filter_mid_m'] = (piezo_meta['top_filter'] + piezo_meta['bottom_filter']) / 200

    pump_meta = pd.read_csv(PUMP_METADATA).rename(columns={'Naam': 'name', 'Xcoor': 'x', 'Ycoor': 'y'})
    river_meta = pd.read_csv(RIVER_METADATA)

    # Piezometer ordering must match training pipeline
    with open(PREPROCESSED_DIR / "column_names_real.txt") as f:
        piezo_names = [l.strip() for l in f if l.strip()]
    piezo_meta = piezo_meta.set_index('name').loc[piezo_names].reset_index()

    def xy_to_rowcol(x, y):
        col = int(round((x - raster_tf.c) / raster_tf.a - 0.5))
        row = int(round((y - raster_tf.f) / raster_tf.e - 0.5))
        return np.clip(row, 0, nrows - 1), np.clip(col, 0, ncols - 1)

    def find_node(x, y, z_mid, prefer_aquifer=False):
        row, col = xy_to_rowcol(x, y)
        present = np.where(valid_mask[:, row, col])[0]
        if len(present) == 0:
            return -1, None

        if np.isnan(z_mid) or prefer_aquifer:
            for li in present:
                if is_aquifer[li]:
                    return node_id_grid[li, row, col], all_layer_names[li]
            return node_id_grid[present[0], row, col], all_layer_names[present[0]]

        for li in present:
            if bots[li, row, col] <= z_mid <= tops[li, row, col]:
                return node_id_grid[li, row, col], all_layer_names[li]

        mids = [(tops[li, row, col] + bots[li, row, col]) / 2 for li in present]
        best = present[np.argmin(np.abs(np.array(mids) - z_mid))]
        return node_id_grid[best, row, col], all_layer_names[best]

    nodes = []
    for _, r in piezo_meta.iterrows():
        nid, layer = find_node(r['x'], r['y'], r['filter_mid_m'])
        nodes.append({'name': r['name'], 'type': 'piezo', 'graph_node': nid, 'layer': layer})

    for _, r in pump_meta.iterrows():
        z_mid = (r['screen_top_nap'] + r['screen_bot_nap']) / 2
        nid, layer = find_node(r['x'], r['y'], z_mid)
        nodes.append({'name': r['name'], 'type': 'pump', 'graph_node': nid, 'layer': layer})

    for _, r in river_meta.iterrows():
        nid, layer = find_node(r['x'], r['y'], np.nan, prefer_aquifer=True)
        nodes.append({'name': r['name'], 'type': 'river', 'graph_node': nid, 'layer': layer})

    node_df = pd.DataFrame(nodes)
    print(f"  Mapped: {len(piezo_meta)} piezo + {len(pump_meta)} pump + {len(river_meta)} river = {len(node_df)}")
    print(f"  Unmapped: {(node_df['graph_node'] == -1).sum()}")
    return node_df


def compute_resistance_matrix(graph, node_df):
    """Run Dijkstra from all infrastructure nodes."""
    n_total = len(node_df)
    graph_ids = node_df['graph_node'].values
    valid = graph_ids >= 0
    valid_ids = graph_ids[valid]

    print(f"  Running Dijkstra from {valid.sum()} sources through {graph.shape[0]:,}-node graph...")
    dist = dijkstra(graph, directed=False, indices=valid_ids)

    resistance = np.full((n_total, n_total), np.inf)
    np.fill_diagonal(resistance, 0)
    valid_idx = np.where(valid)[0]
    for i_out, i_node in enumerate(valid_idx):
        for j_out, j_node in enumerate(valid_idx):
            if i_node != j_node:
                resistance[i_node, j_node] = dist[i_out, valid_ids[j_out]]

    finite = resistance[np.isfinite(resistance) & (resistance > 0)]
    print(f"  Resistance stats: min={finite.min():.1f}, median={np.median(finite):.1f}, max={finite.max():.1f}")
    print(f"  Infinite pairs: {np.sum(np.isinf(resistance) & ~np.eye(n_total, dtype=bool))}")
    return resistance


def main():
    parser = argparse.ArgumentParser(description='Build binary hydraulic resistance matrix from REGIS II')
    parser.add_argument('--full', action='store_true',
                        help='Also generate the full-K resistance matrix (resistance_regis.npy)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: HYDRAULIC_RESISTANCE_DIR from config)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else HYDRAULIC_RESISTANCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"REGIS II directory: {REGIS_DIR}")
    print(f"Output directory: {output_dir}")

    # Discover and load layers
    print("\n1. Loading REGIS II layers...")
    drawable = discover_layers(REGIS_DIR)
    all_names, tops, bots, K_real, is_aquifer, thickness, raster_tf, nrows, ncols, DX = load_layers(drawable)

    n_layers = len(all_names)
    n_aq = is_aquifer.sum()
    n_at = (~is_aquifer).sum()
    print(f"  {n_layers} layers: {n_aq} aquifer + {n_at} aquitard")

    # Map infrastructure nodes (shared across both modes)
    print("\n2. Mapping infrastructure nodes...")
    # We need a valid_mask for node mapping — use real K to determine presence
    valid_real = np.isfinite(thickness) & (thickness > 0) & np.isfinite(K_real) & (K_real > 0)
    node_id_grid_tmp = np.full((n_layers, nrows, ncols), -1, dtype=np.int32)
    node_id_grid_tmp[valid_real] = np.arange(valid_real.sum())
    node_df = map_infrastructure_nodes(
        tops, bots, valid_real, is_aquifer, all_names,
        node_id_grid_tmp, raster_tf, nrows, ncols)

    # Binary resistance matrix
    print("\n3. Building BINARY resistance matrix (K_aquifer={}, K_aquitard={})...".format(K_AQUIFER, K_AQUITARD))
    K_binary = np.full_like(K_real, np.nan)
    for li in range(n_layers):
        if is_aquifer[li]:
            K_binary[li] = np.where(np.isfinite(K_real[li]), K_AQUIFER, np.nan)
        else:
            K_binary[li] = np.where(np.isfinite(K_real[li]), K_AQUITARD, np.nan)

    graph_bin, nid_grid_bin, valid_bin = build_graph(
        tops, bots, thickness, K_binary, is_aquifer, nrows, ncols, DX)

    # Remap infrastructure nodes to binary graph
    # (same spatial locations, but node IDs differ because valid set may differ)
    node_df_bin = node_df.copy()
    for idx, row in node_df_bin.iterrows():
        old_layer = row['layer']
        if old_layer is None:
            continue
        li = all_names.index(old_layer) if old_layer in all_names else -1
        if li >= 0:
            # Find the grid position from the original mapping
            orig_nid = row['graph_node']
            if orig_nid >= 0:
                # Reverse lookup: find (r,c) from original node_id_grid
                positions = np.argwhere(node_id_grid_tmp == orig_nid)
                if len(positions) > 0:
                    _, r, c = positions[0]
                    new_nid = nid_grid_bin[li, r, c]
                    node_df_bin.at[idx, 'graph_node'] = new_nid

    res_binary = compute_resistance_matrix(graph_bin, node_df_bin)
    out_path = output_dir / 'resistance_binary.npy'
    np.save(out_path, res_binary)
    print(f"  Saved: {out_path}")

    # Full-K resistance matrix (optional)
    if args.full:
        print("\n4. Building FULL-K resistance matrix...")
        graph_full, nid_grid_full, valid_full = build_graph(
            tops, bots, thickness, K_real, is_aquifer, nrows, ncols, DX)

        # Remap nodes to full graph
        node_df_full = node_df.copy()
        for idx, row in node_df_full.iterrows():
            old_layer = row['layer']
            if old_layer is None:
                continue
            li = all_names.index(old_layer) if old_layer in all_names else -1
            if li >= 0:
                orig_nid = row['graph_node']
                if orig_nid >= 0:
                    positions = np.argwhere(node_id_grid_tmp == orig_nid)
                    if len(positions) > 0:
                        _, r, c = positions[0]
                        new_nid = nid_grid_full[li, r, c]
                        node_df_full.at[idx, 'graph_node'] = new_nid

        res_full = compute_resistance_matrix(graph_full, node_df_full)
        out_path = output_dir / 'resistance_regis.npy'
        np.save(out_path, res_full)
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
