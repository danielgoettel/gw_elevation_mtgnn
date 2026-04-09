"""
Re-evaluate all v3 models with masked RMSE and reclassify nodes.
Usage: !python /content/gw_elevation_mtgnn/reevaluate_masked.py
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT = Path('/content/drive/MyDrive/Environmental_DL_Project/GroundwaterFlowGNN')
GW_PKG = Path('/content/gw_elevation_mtgnn')
INPUT_DIR = Path('/content/drive/MyDrive/Environmental_DL_Project/GroundwaterFlowGNN-main/data/input')
PREPROC = PROJECT / 'data' / 'preprocessed'
GRAPHS_DIR = PROJECT / 'generated_graphs'
OUTPUTS_DIR = PROJECT / 'outputs' / 'v3'

for p in [str(GW_PKG), str(GW_PKG / 'data_preprocessing'), str(GW_PKG / 'utils'), str(GW_PKG / 'models')]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(GW_PKG)

import config as cfg
cfg.INPUT_DIR = INPUT_DIR
cfg.PREPROCESSED_DIR = PREPROC
cfg.GENERATED_GRAPHS = GRAPHS_DIR
cfg.OUTPUTS_DIR = OUTPUTS_DIR
for attr, subpath in [
    ('PIEZO_METADATA', 'piezometers/piezometer_metadata.csv'),
    ('PUMP_METADATA', 'wells/pump_metadata.csv'),
    ('PUMPING_WELLS_PATH', 'wells/pump_daily.csv'),
    ('PRECIP_PATH', 'meteo_metadata_and_timeseries/precipitation.csv'),
    ('EVAP_PATH', 'meteo_metadata_and_timeseries/evaporation.csv'),
    ('RIVER_PATH', 'river/river_daily.csv'),
    ('PIEZO_LAYER_INFORMATION', 'piezometers/piezometer_layer_information.csv'),
    ('PUMP_DISTANCES', 'wells/wellfield_to_obswell_distances.csv'),
    ('PIEZO_CSV_DIR', 'piezometers/csv/csv'),
    ('RANDOM_FOREST_TRAINING_DATA', 'piezo_only_rf_training_data.csv'),
]:
    setattr(cfg, attr, INPUT_DIR / subpath)
cfg.RF_TRAINED_ALL = PREPROC / 'raw_importances.pkl'
cfg.RF_TRAINED_PIEZOS_ONLY = PREPROC / 'raw_piezo_only_importances.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load data ──
from data_preprocessing import process_data
import data_preprocessing.process_data as _pd
for attr in ['PIEZO_METADATA', 'PUMP_METADATA', 'PUMPING_WELLS_PATH', 'PRECIP_PATH',
             'EVAP_PATH', 'RIVER_PATH', 'PREPROCESSED_DIR', 'INPUT_DIR', 'PIEZO_CSV_DIR',
             'RANDOM_FOREST_TRAINING_DATA', 'PIEZO_LAYER_INFORMATION']:
    setattr(_pd, attr, getattr(cfg, attr))

train_data, val_data, test_data, train_mask, val_mask, test_mask, \
    df_piezo_columns, pump_columns, locations_no_missing, scaler, mean_gw_elevation = \
    process_data.main(synthetic_data=False, resampling_freq='W',
                      val_split=0.667, test_val_size=0.3, legacy_scaling=True)

num_piezo = len(df_piezo_columns)
static_features = torch.load(str(PREPROC / 'static_features.pt'), map_location=device, weights_only=False)

from data_preprocessing.dataset import AutoregressiveTimeSeriesDataset
from models.mtgnn import MTGNN
from utils.training_utils import make_predictions, inverse_transform_with_shape_adjustment

W = 5

# ── Find all completed runs ──
ADJ_PATTERNS = {
    'default': 'adj_default_WM_fixed_piezo_3_pumps_1_*.npy',
    'rf': 'adj_rf_WM_full_piezo_3_pumps_2_*.npy',
    'geolayer': 'adj_geolayer_WM_fixed_piezo_3_pumps_3_*.npy',
    'feature_distance': 'adj_feature_distance_WM_fixed_piezo_3_pumps_4_*.npy',
    'shortest_path': 'adj_shortest_path_WM_fixed_piezo_3_pumps_4_*.npy',
}

all_masked_results = {}

for run_dir in sorted(OUTPUTS_DIR.iterdir()):
    if not run_dir.is_dir() or run_dir.name in ('visualization',):
        continue

    model_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        continue

    for model_dir in model_dirs:
        # Find best fw model
        best_fw = None
        for fw in [1, 2, 3]:
            if (model_dir / f'model_fw{fw}.pt').exists():
                best_fw = fw

        if best_fw is None:
            continue

        model_pt = model_dir / f'model_fw{best_fw}.pt'

        # Determine graph type from run name
        run_name = run_dir.name
        gt = None
        for key in ['default', 'rf', 'geolayer', 'feature_distance', 'shortest_path']:
            if key in run_name:
                gt = key
                break
        if gt is None:
            continue

        # Check if oneway_exo
        one_way = 'oneway_exo' in run_name

        # Load adjacency
        adj_pattern = ADJ_PATTERNS.get(gt)
        if adj_pattern is None:
            continue
        adj_files = sorted(glob.glob(str(GRAPHS_DIR / adj_pattern)))
        if not adj_files:
            continue
        adj = np.load(adj_files[0])
        if one_way:
            adj[num_piezo:, :num_piezo] = 0
            adj[num_piezo:, num_piezo:] = 0
        adj_t = torch.FloatTensor(adj).to(device)

        # Load model
        try:
            model = MTGNN(
                gcn_true=True, build_adj=False, gcn_depth=4,
                num_nodes=adj.shape[0], kernel_set=[1, 2], kernel_size=2,
                dropout=0.5, subgraph_size=20, node_dim=10, dilation_exponential=2,
                conv_channels=64, residual_channels=64, skip_channels=64, end_channels=128,
                seq_length=W + 1, in_dim=1, out_dim=1, layers=4,
                propalpha=0.07, tanhalpha=0.2, layer_norm_affline=True,
                xd=static_features.shape[1], one_way_exo=one_way, num_piezo=num_piezo,
            ).to(device)
            model.load_state_dict(torch.load(str(model_pt), map_location=device, weights_only=False))
            model.eval()
        except Exception as e:
            print(f'  SKIP {run_name}/{model_dir.name}: {e}')
            continue

        # 100-step eval
        test_ds = AutoregressiveTimeSeriesDataset(test_data, W, 100, test_mask, num_piezo)
        test_sample = test_ds[1]
        _, pred_scaled, target_scaled = make_predictions(
            model, test_sample, device, 100, W, adj_t, static_features.to(device),
            num_piezo, build_adj=False, modeltype='MTGNN')

        _, _, _, mask_seq = test_sample
        mask_np = mask_seq.numpy()

        pred_real = inverse_transform_with_shape_adjustment(pred_scaled.numpy(), scaler, num_piezo)
        target_real = inverse_transform_with_shape_adjustment(target_scaled.numpy(), scaler, num_piezo)

        # Masked RMSE per node
        masked_rmses = []
        for i in range(num_piezo):
            m = mask_np[:, i].astype(bool)
            if m.sum() > 0:
                rmse = np.sqrt(np.mean((pred_real[m, i] - target_real[m, i]) ** 2))
            else:
                rmse = np.sqrt(np.mean((pred_real[:, i] - target_real[:, i]) ** 2))
            masked_rmses.append(rmse)

        masked_rmses = np.array(masked_rmses)
        key = f'{run_name}/{model_dir.name}'
        all_masked_results[key] = {
            'run_name': run_name,
            'model_dir': model_dir.name,
            'fw': best_fw,
            'mean_rmse': float(np.nanmean(masked_rmses)),
            'per_node': masked_rmses.tolist(),
        }

        # Save masked RMSE to the model dir
        eval_dir = model_dir / f'eval_fw{best_fw}'
        eval_dir.mkdir(exist_ok=True)
        with open(eval_dir / 'rmse_test_masked.json', 'w') as f:
            json.dump(masked_rmses.tolist(), f)

        print(f'  {run_name:60s}  fw={best_fw}  masked_mean={np.nanmean(masked_rmses):.2f} cm')

# ── Classify nodes using legacy oneway runs ──
legacy_runs = [k for k in all_masked_results
               if 'oneway_exo_legacy_scaling' in k
               and 'dist_limit' not in k and 'gcnd' not in k and 'alpha' not in k]

if legacy_runs:
    stacked = np.array([all_masked_results[k]['per_node'] for k in legacy_runs])
    node_means = np.nanmean(stacked, axis=0)

    if len(legacy_runs) > 1:
        node_cv = np.nanstd(stacked, axis=0) / np.maximum(np.nanmean(stacked, axis=0), 1e-8)
    else:
        node_cv = np.zeros(num_piezo)

    def classify(mean_rmse, cv):
        if cv > 0.15:
            return 'Model-Dependent'
        elif mean_rmse < 10:
            return 'Consistent Low RMSE'
        elif mean_rmse > 30:
            return 'Consistent High RMSE'
        else:
            return 'Consistent Moderate RMSE'

    categories = [classify(node_means[i], node_cv[i]) for i in range(num_piezo)]

    cat_df = pd.DataFrame({
        'name': df_piezo_columns,
        'rmse_mean': node_means,
        'cv': node_cv * 100,
        'category': categories,
    })

    print(f'\n\nNode Classification (masked RMSE):')
    print(cat_df['category'].value_counts())

    print(f'\nTop 10 Consistent High RMSE:')
    high = cat_df[cat_df['category'] == 'Consistent High RMSE'].nlargest(10, 'rmse_mean')
    print(high[['name', 'rmse_mean', 'cv']].to_string(index=False))

    print(f'\nTop 10 Consistent Low RMSE:')
    low = cat_df[cat_df['category'] == 'Consistent Low RMSE'].nsmallest(10, 'rmse_mean')
    print(low[['name', 'rmse_mean', 'cv']].to_string(index=False))

    cat_df.to_csv(OUTPUTS_DIR / 'node_classification_masked.csv', index=False)
    print(f'\nSaved to {OUTPUTS_DIR / "node_classification_masked.csv"}')
else:
    print('\nNo legacy oneway runs found for classification')

# Save all results
with open(OUTPUTS_DIR / 'all_masked_results.json', 'w') as f:
    json.dump({k: {'run_name': v['run_name'], 'fw': v['fw'], 'mean_rmse': v['mean_rmse']}
               for k, v in all_masked_results.items()}, f, indent=2)

print(f'\nDone! Re-evaluated {len(all_masked_results)} runs with masked RMSE.')
