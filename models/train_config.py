from pathlib import Path

def define_base_configuration():
    return {
        'synthetic_data': False,
        'percentage': 100,
        'n_piezo_connected': 3,
        'W': 5,
        'gcn_true': True,
        'build_adj': False,
        'gcn_depth': 4,
        'kernel_set': [1, 2],
        'kernel_size': 2,
        'dropout': 0.5,
        'subgraph_size': 20,
        'node_dim': 10,
        'dilation_exponential': 2,
        'conv_channels': 64,
        'residual_channels': 64,
        'skip_channels': 64,
        'end_channels': 128,
        'in_dim': 1,  # Not varied
        'out_dim': 1,  # Not varied
        'layers': 4,
        'propalpha': 0.07,
        'tanhalpha': 0.2,
        'layer_norm_affline': True,
        'graph_type' :'rf',
        'feature_importance_multiplier' : 1,
        'n_pumps_connected' : 4,
        'weight_mode': 'fixed',  # 'fixed' or 'variable'
        'layer_constrain' : False,
        'perturb_weights' : False,
        'exclude_evap_precip' : "Garg",

        # Training parameters
        'learning_rate': 0.001,
        'num_epochs': 200,
        'batch_size': 32,
        'F_w': 3,
        'model_type': 'MTGNN',  # Options: 'MTGNN', 'LSTM'
        'early_stopping_patience': 30,
        'min_delta': 0.001,
        'scheduler_patience': 10,

        # Data split — 70/10/20 train/val/test
        'test_val_size': 0.3,
        'val_split': 0.667,

        # Directed graph — piezo-piezo edges flow from higher to lower GW elevation
        'directed_graph': False,

        # Dynamic node dropout — isolate high-RMSE nodes mid-training
        'node_dropout': False,
        'node_dropout_warmup': 40,          # epochs before first dropout evaluation
        'node_dropout_sd_threshold': 3.0,   # drop nodes with RMSE > mean + threshold * SD
        'node_dropout_eval_steps': 20,      # autoregressive steps for dropout RMSE (detects compounding errors)
        'node_dropout_check_interval': 20,  # re-check every N epochs after warmup

        # RF variable weight range — scales RF importances to [min, max]
        'rf_weight_min': 0.08,
        'rf_weight_max': 0.2,

        # RF full VIM mode — minimum importance threshold (weight_mode='full')
        'rf_vim_min': 0.01,
        'rf_min_connections': 3,   # Minimum piezo-piezo connections in full VIM mode

        # Data resampling — 'W' for weekly, 'D' for daily, None for native 3-hourly
        'resampling_freq': 'W',

        # Multi-support — Graph WaveNet-style parallel static + adaptive adjacency
        'multi_support': False,
        'adaptive_graph_type': 'rf',       # Source for adaptive init: 'rf', 'geolayer', 'default'
        'adaptive_weight_mode': 'fixed',   # Weight mode for adaptive init (when adaptive_graph_type='rf')

        # Mixed-optimal graph — path to the RMSE table used to select best variant per node
        # Set to None to use default: OUTPUTS_DIR / "per_node_rmse_all_variants.xlsx"
        'rmse_table_path': None,

        # Shortest-path graph (graph_type='shortest_path') — uses pre-computed
        # Dijkstra resistance matrices from REGIS II subsurface data
        'sp_config': {
            'resistance_source': 'regis',    # 'regis' (full K) or 'binary' (aquifer/aquitard)
            'sp_min_sensitivity': 0.0,       # min connectivity threshold (0 = use top-N only)
            'sp_min_connections': 3,         # guaranteed min piezo-piezo connections
            'piezo_weight_range': (0.08, 0.2),
            'pump_weight_range': (0.15, 0.25),
            'river_weight_range': (0.4, 0.6),
            'n_rivers_connected': 2,
            'include_pumps': True,
            'include_rivers': True,
            'use_hydraulic_exo': False,      # True = use resistance for pump/river weights
        },

        # Feature-distance graph (graph_type='feature_distance') — Liang et al. 2025
        # 7-D feature-space distance: [X, Y, Z, log10(Kh), log10(Kh), log10(Kv), h]
        'fd_config': {
            'radius': 0.25,            # max distance in normalised feature space for edge creation
            'min_connections': 3,      # guaranteed minimum connections per node
        },

        # RF cutoff graph (graph_type='rf', weight_mode='cutoff')
        # Uses full RF importance matrix (incl. pumps & rivers) with threshold cutoff
        'rf_config': {
            'cutoff': 0.01,            # min RF importance to create an edge
            'min_connections': 3,      # guaranteed minimum piezo-piezo connections
        },

        # Log-scaled pump weights (Thiem equation) — set to a dict to enable.
        # Distance-based pump weights: closer pump → higher weight.
        # 'multiplier' tunes the scaling (1.0 = full range, <1 compresses to w_min).
        # Weights are clipped to [w_min, w_max].
        # 'R' is the radius of influence in metres (default 25 km).
        'log_pump_config': None,
        # Example: {'multiplier': 1.0, 'w_min': 0.1, 'w_max': 0.3, 'R': 25000}
    }

# --------------------------------------------------------------------------
# Explicit configuration list — each dict overrides base_config for that run.
# When this list is non-empty, parameter_variations is ignored.
# --------------------------------------------------------------------------
_SEEDS = [42, 123, 256, 512, 777, 1024, 2048, 3141]
_OUTPUT_TAG = 'v3'

# ── Per-variant base overrides (graph-type-specific settings) ──
_VARIANT_BASES = {
    'default': {'graph_type': 'default'},
    'geolayer': {'graph_type': 'geolayer'},
    'rf_vim002': {
        'graph_type': 'rf',
        'weight_mode': 'full',
        'rf_vim_min': 0.02,
        'rf_weight_min': 0.08,
        'rf_weight_max': 0.2,
    },
    'fd_r015_wm02': {
        'graph_type': 'feature_distance',
        'fd_config': {'radius': 0.15, 'weight_max': 0.2},
    },
    'sp_binary_hexo': {
        'graph_type': 'shortest_path',
        'sp_config': {
            'resistance_source': 'binary',
            'sp_min_sensitivity': 0.005,
            'sp_min_connections': 2,
            'use_hydraulic_exo': True,
        },
    },
    'mixed': {'graph_type': 'mixed'},
    'rf_ms': {
        'graph_type': 'rf',
        'multi_support': True,
        'build_adj': False,
        'adaptive_graph_type': 'rf',
    },
    'adaptive': {
        'graph_type': 'default',
        'build_adj': True,
        'gcn_true': True,
    },
}

# ── 5 new conditions: n_pumps × dropout ──
_CONDITIONS = [
    {'n_pumps_connected': 2, 'node_dropout': False},   # n_pumps=2
    {'n_pumps_connected': 1, 'node_dropout': False},   # n_pumps=1
    {'n_pumps_connected': 3, 'node_dropout': True},    # n_pumps=3 + dropout
    {'n_pumps_connected': 2, 'node_dropout': True},    # n_pumps=2 + dropout
    {'n_pumps_connected': 1, 'node_dropout': True},    # n_pumps=1 + dropout
]

# ── Best config per graph family (from seed experiment analysis) ──
# Each entry uses the n_pumps and dropout settings that produced the lowest RMSE.
_ABLATION_SEEDS = [42, 123, 256]

_BEST_PER_FAMILY = {
    'default': {
        'graph_type': 'default',
        'n_pumps_connected': 1, 'node_dropout': False,
    },
    'geolayer': {
        'graph_type': 'geolayer',
        'n_pumps_connected': 3, 'node_dropout': False,
    },
    'rf': {
        'graph_type': 'rf', 'weight_mode': 'full',
        'rf_vim_min': 0.02, 'rf_weight_min': 0.08, 'rf_weight_max': 0.2,
        'n_pumps_connected': 2, 'node_dropout': True,
    },
    'feature_distance': {
        'graph_type': 'feature_distance',
        'fd_config': {'radius': 0.15, 'weight_max': 0.2},
        'n_pumps_connected': 4, 'node_dropout': False,
    },
    'sp_binary': {
        'graph_type': 'shortest_path',
        'sp_config': {
            'resistance_source': 'binary',
            'sp_min_sensitivity': 0.005,
            'sp_min_connections': 2,
            'use_hydraulic_exo': True,
        },
        'n_pumps_connected': 4, 'node_dropout': False,
    },
    'sp_regis': {
        'graph_type': 'shortest_path',
        'sp_config': {
            'resistance_source': 'regis',
            'sp_min_sensitivity': 0.02,
            'sp_min_connections': 2,
        },
        'n_pumps_connected': 4, 'node_dropout': False,
    },
    'adaptive': {
        'graph_type': 'default',
        'build_adj': True, 'gcn_true': True,
        'n_pumps_connected': 4, 'node_dropout': False,
    },
}

# ── Exogenous ablation conditions ──
_EXO_ABLATIONS = [
    # Baselines (no ablation) already exist in 5_rivers/ with 8 seeds
    {'remove_pumps': True},                                      # no pumps
    {'remove_rivers': True},                                     # no rivers
    {'remove_pumps': True, 'remove_rivers': True},               # no pumps or rivers
    {'remove_precip': True, 'remove_evap': True},                # no climate forcing
    {'remove_pumps': True, 'remove_rivers': True,
     'remove_precip': True, 'remove_evap': True},                # piezo-only
]

# ── Pump weight ablation: Thiem vs Coherence × 7 families × 3 seeds = 42 runs ──
_PUMP_WEIGHT_FAMILIES = {k: v for k, v in _BEST_PER_FAMILY.items() if k != 'adaptive'}

_PUMP_WEIGHT_SCHEMES = {
    'thiem': {'source': 'thiem', 'multiplier': 1.0, 'w_min': 0.1, 'w_max': 0.3, 'R': 10000},
    'coherence': {'source': 'coherence'},
    'coherence_90_365d': {'source': 'coherence', 'band': '90_365d'},
    'coherence_gt365d': {'source': 'coherence', 'band': 'gt365d'},
}

_EXTEND_SEEDS = [512, 777, 1024, 2048, 3141]

# ── Configs that scored <20 cm mean on 3 seeds → extend to 8 seeds ──
# Maps ablation condition to the exo_ablation override dict.
_EXO_COND_MAP = {
    'no_pumps': {'remove_pumps': True},
    'no_rivers': {'remove_rivers': True},
    'no_pumps_no_rivers': {'remove_pumps': True, 'remove_rivers': True},
    'no_evap_no_precip': {'remove_precip': True, 'remove_evap': True},
    'no_evap_no_precip_no_pumps_no_rivers': {
        'remove_pumps': True, 'remove_rivers': True,
        'remove_precip': True, 'remove_evap': True},
}

# ── Thiem radius ablation: 3-seed exploratory runs ──
# The normalization was fixed to use actual r_min instead of 1m,
# so weights now span the full [w_min, w_max] range.
_THIEM_ABLATION_SEEDS = [42, 123, 256]
_PUMP_WEIGHT_FAMILIES_NO_ADAPTIVE = {k: v for k, v in _BEST_PER_FAMILY.items() if k != 'adaptive'}

# R=10km with fixed normalization (7 families × 3 seeds = 21 runs)
_THIEM_R10_FIXED = {
    'source': 'thiem', 'multiplier': 1.0, 'w_min': 0.1, 'w_max': 0.3, 'R': 10000,
}
# R=5km (7 families × 3 seeds = 21 runs)
_THIEM_R5 = {
    'source': 'thiem', 'multiplier': 1.0, 'w_min': 0.1, 'w_max': 0.3, 'R': 5000,
}
# R=15km (7 families × 3 seeds = 21 runs)
_THIEM_R15 = {
    'source': 'thiem', 'multiplier': 1.0, 'w_min': 0.1, 'w_max': 0.3, 'R': 15000,
}

# ── Coherence cutoff ablation: 3-seed exploratory runs ──
# Current threshold ≈ 0.05 (95% significance). Tighter = fewer but stronger edges.
# Broadband (max sub-annual): t=0.05 → 637 edges, t=0.08 → 378, t=0.10 → 283
# 90-365d:                     t=0.05 → 526,       t=0.08 → 362, t=0.10 → 276
# >365d:                       t=0.05 → 657,       t=0.08 → 499, t=0.10 → 432
_COH_TIGHT_SCHEMES = {
    'coh_t008':         {'source': 'coherence', 'band': 't008'},
    'coh_t010':         {'source': 'coherence', 'band': 't010'},
    'coh_90_365d_t008': {'source': 'coherence', 'band': '90_365d_t008'},
    'coh_90_365d_t010': {'source': 'coherence', 'band': '90_365d_t010'},
    'coh_gt365d_t008':  {'source': 'coherence', 'band': 'gt365d_t008'},
    'coh_gt365d_t010':  {'source': 'coherence', 'band': 'gt365d_t010'},
}

# ── Fill the Family × Condition grid to 8 seeds ──
# Grid fill — 1 cell remaining (SP-binary thiem 5 seeds).
# RF, SP-REGIS, adaptive exo all complete. SP-binary thiem at 3/8.

_ALL_SEEDS = [42, 123, 256, 512, 777, 1024, 2048, 3141]

_GRID_FILL = {
    # ── RF: COMPLETE (4/4 cells × 8 seeds) — removed ──
    # ── Adaptive exo: COMPLETE (5/5 cells × 8 seeds) — removed ──
    # ── SP-REGIS: COMPLETE (6/6 cells × 8 seeds) — removed ──
    # ── SP-binary: 5/6 cells complete ──
    # ('sp_binary', 'no_rivers'):                        COMPLETE — removed
    # ('sp_binary', 'no_pumps_no_rivers'):               COMPLETE — removed
    # ('sp_binary', 'no_evap_no_precip'):                COMPLETE — removed
    # ('sp_binary', 'no_evap_no_precip_no_pumps_no_rivers'): COMPLETE — removed
    # ('sp_binary', 'coherence_gt365d'):                 COMPLETE — removed
    ('sp_binary', 'thiem'):                             _EXTEND_SEEDS,
}

def _build_grid_configs():
    """Generate explicit_configs for v3 runs."""
    configs = []

    _ALL_SEEDS = [42, 123, 256, 512, 777, 1024, 2048, 3141]
    _SEED = 42

    # ══════════════════════════════════════════════════════════════════
    # GCN depth ablation: default graph, gcn_depth=2 × layers=3,4
    # Reduces effective multi-hop reach from 16 to 6-8 hops
    # ══════════════════════════════════════════════════════════════════
    for n_layers in [3, 4]:
        configs.append({
            **_BEST_PER_FAMILY['default'],
            'legacy_scaling': True,
            'one_way_exo': True,
            'gcn_depth': 2,
            'layers': n_layers,
            'seed': _SEED,
            'seed_experiment_name': _OUTPUT_TAG,
        })

    return configs

explicit_configs = _build_grid_configs()

# Define variations for each parameter (Cartesian product — only used when explicit_configs is empty)
parameter_variations = {}

