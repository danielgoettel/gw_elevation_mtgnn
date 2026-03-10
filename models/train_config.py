def define_base_configuration():
    return {
        'synthetic_data': False,
        'percentage': 100,
        'n_piezo_connected': 3,
        'W': 5,
        'gcn_true': True,
        'build_adj': False,  #ALWAYS FALSE
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
    }

# --------------------------------------------------------------------------
# Explicit configuration list — each dict overrides base_config for that run.
# When this list is non-empty, parameter_variations is ignored.
# --------------------------------------------------------------------------
_DROP_NODE = ['B39F0739-003']
_SEEDS = [42, 123, 256, 512, 777, 1024, 2048, 3141]
_5R_TAG = 'seed_experiment/5_rivers'

_DO_TAG = 'seed_experiment/5_rivers'  # same output folder, dropout suffix in filename

explicit_configs = [
    # ══════════════════════════════════════════════════════════════════
    # Fill remaining baseline gaps (pump=4, no dropout)
    # ══════════════════════════════════════════════════════════════════
    # ---- FD r=0.15 wm=0.2 baseline: 5 missing seeds ----
    *[{'graph_type': 'feature_distance',
       'fd_config': {'radius': 0.15, 'weight_max': 0.2},
       'seed': s, 'seed_experiment_name': _5R_TAG}
      for s in [512, 777, 1024, 2048, 3141]],
    # ---- Mixed baseline: all 8 seeds ----
    *[{'graph_type': 'mixed',
       'seed': s, 'seed_experiment_name': _5R_TAG}
      for s in _SEEDS],
]

# Previous config (kept for reference):
# explicit_configs = [
#     # ---- Drop high-RMSE node: default, geolayer, rf x 8 seeds ----
#     *[{'graph_type': gt, 'seed': s, 'exclude_nodes': _DROP_NODE}
#       for gt in ('default', 'geolayer', 'rf')
#       for s in _SEEDS],
# ]

# Previous config (kept for reference):
# explicit_configs = [
#     # ---- Mixed-optimal graph x 8 seeds ----
#     {'graph_type': 'mixed', 'seed': 42},
#     {'graph_type': 'mixed', 'seed': 123},
#     {'graph_type': 'mixed', 'seed': 256},
#     {'graph_type': 'mixed', 'seed': 512},
#     {'graph_type': 'mixed', 'seed': 777},
#     {'graph_type': 'mixed', 'seed': 1024},
#     {'graph_type': 'mixed', 'seed': 2048},
#     {'graph_type': 'mixed', 'seed': 3141},
# ]

# Define variations for each parameter (Cartesian product — only used when explicit_configs is empty)
parameter_variations = {
  #'graph_type': ['default', 'geolayer', 'rf'],
  #'node_dropout': [True, False],
  #'feature_importance_multiplier' : [0.1],
  #'n_piezo_connected' : [4,6],
  #'n_pumps_connected' : [3],
  #'layer_constrain' : [False],
  #'weight_mode' : ['fixed']
  #'skip_channels': [128, 256],
  #'W': [15],
  #'kernel_set': [[2, 3], [3, 4]],
  #'kernel_size': [2, 3],
  #'dilation exponential' : [1,2,3]
}
