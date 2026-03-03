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
        'ext_data' : True,
        'multiply_exo_weights' : False,
        'layer_constrain' : False,
        'perturb_weights' : False,
        'exclude_evap_precip' : "Garg",

        # Training parameters
        'learning_rate': 0.001,
        'num_epochs': 200,
        'batch_size': 32,
        'F_w': 1,
        'model_type': 'MTGNN',  # Options: 'MTGNN', 'MultigraphGNN', 'LSTM'
        'early_stopping_patience': 30,
        'min_delta': 0.001,
        'scheduler_patience': 10,

        # MultigraphGNN parameters (only used when model_type='MultigraphGNN')
        'num_relations': 5,
        'rgcn_num_bases': None,  # basis decomposition (None = no decomposition)

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

        # Data resampling — 'W' for weekly, 'D' for daily, None for native 3-hourly
        'resampling_freq': 'W',
    }

# --------------------------------------------------------------------------
# Explicit configuration list — each dict overrides base_config for that run.
# When this list is non-empty, parameter_variations is ignored.
# --------------------------------------------------------------------------
explicit_configs = [
    # ---- F_w=1 vanilla re-run ----
    {'graph_type': 'default'},
    # ---- F_w=1 RF directed re-runs with n_piezo=6 ----
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'directed_graph': True, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'directed_graph': True, 'n_piezo_connected': 6},
    # ---- F_w=3 directed runs with n_piezo=6 ----
    {'graph_type': 'default', 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    {'graph_type': 'geolayer', 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'layer_constrain': True, 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.05, 'rf_weight_max': 0.3, 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'layer_constrain': True, 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'directed_graph': True, 'F_w': 3, 'n_piezo_connected': 6},
    # ---- F_w=5 undirected runs with n_piezo=3 ----
    {'graph_type': 'default', 'F_w': 5},
    {'graph_type': 'geolayer', 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'layer_constrain': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'layer_constrain': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.05, 'rf_weight_max': 0.3, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'layer_constrain': True, 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'layer_constrain': True, 'rf_weight_min': 0.05, 'rf_weight_max': 0.3, 'F_w': 5},
    # ---- F_w=5 directed runs with n_piezo=3 ----
    {'graph_type': 'default', 'directed_graph': True, 'F_w': 5},
    {'graph_type': 'geolayer', 'directed_graph': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'directed_graph': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'layer_constrain': True, 'directed_graph': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'directed_graph': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.05, 'rf_weight_max': 0.3, 'directed_graph': True, 'F_w': 5},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'layer_constrain': True, 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'directed_graph': True, 'F_w': 5},
    # ---- F_w=5 directed runs with n_piezo=6 ----
    {'graph_type': 'default', 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
    {'graph_type': 'geolayer', 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'fixed', 'layer_constrain': True, 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'rf_weight_min': 0.05, 'rf_weight_max': 0.3, 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
    {'graph_type': 'rf', 'weight_mode': 'variable', 'layer_constrain': True, 'rf_weight_min': 0.08, 'rf_weight_max': 0.2, 'directed_graph': True, 'F_w': 5, 'n_piezo_connected': 6},
]

# Define variations for each parameter (Cartesian product — only used when explicit_configs is empty)
parameter_variations = {
  #'graph_type': ['default', 'geolayer', 'rf'],
  #'node_dropout': [True, False],
  #'feature_importance_multiplier' : [0.1],
  #'n_piezo_connected' : [4,6],
  #'n_pumps_connected' : [3],
  #'layer_constrain' : [False],
  #'multiply_exo_weights' : [False],
  #'weight_mode' : ['fixed']
  #'skip_channels': [128, 256],
  #'W': [15],
  #'kernel_set': [[2, 3], [3, 4]],
  #'kernel_size': [2, 3],
  #'dilation exponential' : [1,2,3]
}
