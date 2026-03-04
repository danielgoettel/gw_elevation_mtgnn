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
        'F_w': 1,
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

        # Data resampling — 'W' for weekly, 'D' for daily, None for native 3-hourly
        'resampling_freq': 'W',

        # Multi-support — Graph WaveNet-style parallel static + adaptive adjacency
        'multi_support': False,
        'adaptive_graph_type': 'rf',       # Source for adaptive init: 'rf', 'geolayer', 'default'
        'adaptive_weight_mode': 'fixed',   # Weight mode for adaptive init (when adaptive_graph_type='rf')
    }

# --------------------------------------------------------------------------
# Explicit configuration list — each dict overrides base_config for that run.
# When this list is non-empty, parameter_variations is ignored.
# --------------------------------------------------------------------------
explicit_configs = [
    # ---- Daily runs: rf variable, geolayer ----
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'resampling_freq': 'D', 'F_w': 2, 'early_stopping_patience': 20, 'batch_size': 128},
    {'graph_type': 'geolayer', 'resampling_freq': 'D', 'F_w': 2, 'early_stopping_patience': 20, 'batch_size': 128},
]

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
