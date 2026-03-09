"""
Extract per-node hydraulic conductivity (Kh, Kv) from REGIS II rasters and
compute the full 7-D feature-distance matrix (Liang et al. 2025).

For each infrastructure node (piezometer, pump, river station), this script:
  1. Locates the node in the REGIS II 3-D grid using its (x, y, z) coordinates
  2. Extracts the real K value from the layer the node belongs to
  3. Derives both Kh and Kv:
     - Aquifer layer  → Kh = layer kh, Kv = nearest aquitard kv
     - Aquitard layer → Kv = layer kv, Kh = nearest aquifer kh
  4. Builds 7-D feature vectors [X, Y, Z, log10(Kh), log10(Kh), log10(Kv), h]
  5. Computes the 209×209 pairwise Euclidean distance matrix in normalised space

Outputs:
  node_k_values.csv         — columns: name, type, kh, kv
  feature_distance_7d.npy   — 209×209 pairwise distance matrix

Usage:
    python -m data_preprocessing.extract_node_k_values
"""

import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    INPUT_DIR, PREPROCESSED_DIR, REGIS_DIR,
    PIEZO_METADATA, PUMP_METADATA, RIVER_METADATA,
    NODE_K_VALUES, FEATURE_DISTANCE_7D, PROCESSED_DATA_FILE,
)
from data_preprocessing.build_binary_resistance import (
    discover_layers, load_layers,
)


def calculate_7d_matrix():
    """Extract K values from REGIS II and compute the 7-D feature-distance matrix."""

    # ── 1. Load REGIS II layers ──────────────────────────────────────────
    print("1. Loading REGIS II layers...")
    drawable = discover_layers(REGIS_DIR)
    all_names, tops, bots, K_real, is_aquifer, thickness, raster_tf, nrows, ncols, DX = load_layers(drawable)
    n_layers = len(all_names)
    print(f"   {n_layers} layers: {is_aquifer.sum()} aquifer + {(~is_aquifer).sum()} aquitard")

    valid_mask = np.isfinite(thickness) & (thickness > 0) & np.isfinite(K_real) & (K_real > 0)

    # ── 2. Load node metadata ────────────────────────────────────────────
    print("2. Loading node metadata...")

    # Piezometers — must match training pipeline order
    piezo_meta = pd.read_csv(PIEZO_METADATA)
    piezo_layers = pd.read_csv(INPUT_DIR / "piezometers" / "piezometer_layer_information.csv")
    piezo_meta = piezo_meta.merge(piezo_layers, on='name', how='left')
    piezo_meta['filter_mid_m'] = (piezo_meta['top_filter'] + piezo_meta['bottom_filter']) / 200

    with open(PREPROCESSED_DIR / "column_names_real.txt") as f:
        piezo_names = [line.strip() for line in f if line.strip()]
    piezo_meta = piezo_meta.set_index('name').loc[piezo_names].reset_index()

    # Pumps
    pump_meta = pd.read_csv(PUMP_METADATA).rename(
        columns={'Naam': 'name', 'Xcoor': 'x', 'Ycoor': 'y'})

    # Rivers
    river_meta = pd.read_csv(RIVER_METADATA)

    # ── 3. Helper functions (reuse patterns from build_binary_resistance) ─
    def xy_to_rowcol(x, y):
        col = int(round((x - raster_tf.c) / raster_tf.a - 0.5))
        row = int(round((y - raster_tf.f) / raster_tf.e - 0.5))
        return np.clip(row, 0, nrows - 1), np.clip(col, 0, ncols - 1)

    def find_layer_index(x, y, z_mid, prefer_aquifer=False):
        """Return (layer_index, row, col) for a node, or (-1, r, c) if unmapped."""
        row, col = xy_to_rowcol(x, y)
        present = np.where(valid_mask[:, row, col])[0]
        if len(present) == 0:
            return -1, row, col

        if np.isnan(z_mid) or prefer_aquifer:
            for li in present:
                if is_aquifer[li]:
                    return li, row, col
            return present[0], row, col

        # Match by depth
        for li in present:
            if bots[li, row, col] <= z_mid <= tops[li, row, col]:
                return li, row, col

        # Closest layer midpoint
        mids = [(tops[li, row, col] + bots[li, row, col]) / 2 for li in present]
        best = present[np.argmin(np.abs(np.array(mids) - z_mid))]
        return best, row, col

    def get_kh_kv(li, row, col):
        """Derive both Kh and Kv for a node at (layer, row, col).

        If the node's layer is an aquifer  → Kh from that layer, Kv from nearest aquitard.
        If the node's layer is an aquitard → Kv from that layer, Kh from nearest aquifer.
        """
        if li < 0:
            return np.nan, np.nan

        k_self = K_real[li, row, col]
        if not np.isfinite(k_self) or k_self <= 0:
            # Fallback: layer median
            layer_vals = K_real[li][np.isfinite(K_real[li]) & (K_real[li] > 0)]
            k_self = np.median(layer_vals) if len(layer_vals) > 0 else np.nan

        if is_aquifer[li]:
            kh = k_self
            # Find nearest aquitard for Kv
            kv = _nearest_other_k(li, row, col, want_aquifer=False)
        else:
            kv = k_self
            # Find nearest aquifer for Kh
            kh = _nearest_other_k(li, row, col, want_aquifer=True)

        return kh, kv

    def _nearest_other_k(li, row, col, want_aquifer):
        """Get K from the nearest layer of the opposite type (aquifer/aquitard)."""
        # Search outward from current layer
        for offset in range(1, n_layers):
            for candidate in [li - offset, li + offset]:
                if 0 <= candidate < n_layers and is_aquifer[candidate] == want_aquifer:
                    if valid_mask[candidate, row, col]:
                        k = K_real[candidate, row, col]
                        if np.isfinite(k) and k > 0:
                            return k
        # Fallback: median of all layers of that type at this (row, col)
        mask_type = is_aquifer if want_aquifer else ~is_aquifer
        vals = K_real[mask_type, row, col]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) > 0:
            return np.median(vals)
        # Global fallback: median across entire grid for that type
        all_vals = K_real[mask_type]
        all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        return np.median(all_vals) if len(all_vals) > 0 else np.nan

    # ── 4. Extract K values for each node ────────────────────────────────
    print("3. Extracting K values...")
    records = []

    # Piezometers
    for _, r in piezo_meta.iterrows():
        li, row, col = find_layer_index(r['x'], r['y'], r['filter_mid_m'])
        kh, kv = get_kh_kv(li, row, col)
        records.append({'name': r['name'], 'type': 'piezo', 'kh': kh, 'kv': kv,
                        'x': r['x'], 'y': r['y'], 'z': r['filter_mid_m'],
                        'layer': all_names[li] if li >= 0 else None})

    # Pumps
    for _, r in pump_meta.iterrows():
        z_mid = (r['screen_top_nap'] + r['screen_bot_nap']) / 2
        li, row, col = find_layer_index(r['x'], r['y'], z_mid)
        kh, kv = get_kh_kv(li, row, col)
        records.append({'name': r['name'], 'type': 'pump', 'kh': kh, 'kv': kv,
                        'x': r['x'], 'y': r['y'], 'z': z_mid,
                        'layer': all_names[li] if li >= 0 else None})

    # Rivers (prefer aquifer, surface)
    for _, r in river_meta.iterrows():
        li, row, col = find_layer_index(r['x'], r['y'], np.nan, prefer_aquifer=True)
        kh, kv = get_kh_kv(li, row, col)
        records.append({'name': r['name'], 'type': 'river', 'kh': kh, 'kv': kv,
                        'x': r['x'], 'y': r['y'], 'z': r.get('z', 0.0),
                        'layer': all_names[li] if li >= 0 else None})

    result_df = pd.DataFrame(records)

    # ── 5. Summary and save K values ─────────────────────────────────────
    n_piezo = (result_df['type'] == 'piezo').sum()
    n_pump = (result_df['type'] == 'pump').sum()
    n_river = (result_df['type'] == 'river').sum()
    n_nan = result_df[['kh', 'kv']].isna().any(axis=1).sum()

    print(f"   Nodes: {n_piezo} piezo + {n_pump} pump + {n_river} river = {len(result_df)}")
    print(f"   Kh range: {result_df['kh'].min():.4f} – {result_df['kh'].max():.4f} m/d")
    print(f"   Kv range: {result_df['kv'].min():.6f} – {result_df['kv'].max():.6f} m/d")
    print(f"   Nodes with NaN: {n_nan}")

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_df = result_df[['name', 'type', 'kh', 'kv']]
    out_df.to_csv(NODE_K_VALUES, index=False)
    print(f"\n   Saved K values: {NODE_K_VALUES}")

    # Also save with layer info for inspection
    diag_path = PREPROCESSED_DIR / "node_k_values_with_layers.csv"
    result_df.to_csv(diag_path, index=False)
    print(f"   Diagnostics: {diag_path}")

    # ── 6. Compute 7-D feature-distance matrix ───────────────────────────
    print("\n4. Computing 7-D feature-distance matrix...")

    # Load mean groundwater elevation from processed data
    print("   Loading mean GW elevation from processed_data.pkl...")
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        df_piezo, _ = pickle.load(f)
    mean_gw_elevation = df_piezo.mean().values  # (num_piezo,)
    print(f"   Mean GW elevation: {mean_gw_elevation.min():.2f} – {mean_gw_elevation.max():.2f} m NAP")

    # Build 7-D feature matrix: [X, Y, Z, log10(Kh), log10(Kh), log10(Kv), h]
    n_nodes = len(result_df)
    features = np.zeros((n_nodes, 7))

    for i, row in result_df.iterrows():
        features[i, 0] = row['x']
        features[i, 1] = row['y']
        features[i, 2] = row['z']
        features[i, 3] = np.log10(max(row['kh'], 1e-10))  # log10(Kx)
        features[i, 4] = np.log10(max(row['kh'], 1e-10))  # log10(Ky) = log10(Kx) (isotropy)
        features[i, 5] = np.log10(max(row['kv'], 1e-10))  # log10(Kz)

        # Steady-state head
        if row['type'] == 'piezo':
            features[i, 6] = mean_gw_elevation[i]
        else:
            # Pumps/rivers: use Z as proxy
            features[i, 6] = row['z']

    # Min-max normalise each dimension to [0, 1]
    feat_min = features.min(axis=0)
    feat_max = features.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    features_norm = (features - feat_min) / feat_range

    # Pairwise Euclidean distance
    dist_7d = cdist(features_norm, features_norm, metric='euclidean')

    np.save(FEATURE_DISTANCE_7D, dist_7d)
    print(f"   Distance matrix shape: {dist_7d.shape}")
    finite_nonzero = dist_7d[dist_7d > 0]
    print(f"   Distance range: {finite_nonzero.min():.4f} – {finite_nonzero.max():.4f}")
    print(f"   Saved: {FEATURE_DISTANCE_7D}")

    # Also save as CSV with node names as row/column labels
    node_names = result_df['name'].tolist()
    csv_path = FEATURE_DISTANCE_7D.with_suffix('.csv')
    dist_df = pd.DataFrame(dist_7d, index=node_names, columns=node_names)
    dist_df.to_csv(csv_path)
    print(f"   Saved CSV: {csv_path}")

    print("\nDone!")


if __name__ == '__main__':
    calculate_7d_matrix()
