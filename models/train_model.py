import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler 
from torch.amp import autocast
import json

import itertools
import sys

from data_preprocessing import process_data
from data_preprocessing import gnn_data_prep
from models.mtgnn import MTGNN
from models.multigraph_gnn import MultigraphGNN

from models.lstm_model import LSTMModel
from data_preprocessing.dataset import AutoregressiveTimeSeriesDataset
from utils.training_utils import prepare_combined_input, make_predictions, inverse_transform_with_shape_adjustment, generate_model_filename, save_rmse_values, record_result, analyze_results, get_synthetic

from utils.metrics import calculate_rmse_per_piezometer, calculate_rmse_per_piezometer_moria, print_mean_std
from utils.visualization import plot_sequences, plot_sparsity_pattern, plot_comparison_sequence, plot_comparison_sequence_dual_y, plot_rmse_comparison, plot_rmse_3d_network, plot_adj_heatmap

from config import PIEZO_LAYER_INFORMATION, RANDOM_FOREST_TRAINING_DATA, SCATTER_PLOTS, TRAINING_SUMMARIES, SAVED_MODELS_DIR, TRAINING_RESULTS_DIR, RUN_PLOTS_AND_RESULTS, OUTPUTS_DIR

from train_config import define_base_configuration, parameter_variations

import time
import datetime




from functools import partial


def create_model(num_features, num_nodes, seq_length, model_type, **kwargs):
    """Factory function to create the appropriate model based on model_type."""
    if model_type == 'MTGNN':
        return _create_mtgnn(num_features, num_nodes, seq_length, **kwargs)
    elif model_type == 'MultigraphGNN':
        return _create_multigraph_gnn(num_nodes, seq_length, **kwargs)
    elif model_type == 'LSTM':
        return _create_lstm_model()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _create_mtgnn(num_features, num_nodes, seq_length, **kwargs):
    """Create MTGNN model."""
    mtgnn_params = {key: kwargs[key] for key in kwargs if key in {
        'gcn_true', 'build_adj', 'gcn_depth', 'kernel_set', 'kernel_size',
        'dropout', 'subgraph_size', 'node_dim', 'dilation_exponential',
        'conv_channels', 'residual_channels', 'skip_channels', 'end_channels',
        'in_dim', 'out_dim', 'layers', 'propalpha', 'tanhalpha',
        'layer_norm_affline', 'xd'
    }}
    mtgnn_params['num_nodes'] = num_nodes
    mtgnn_params['seq_length'] = seq_length + 1
    mtgnn_params['in_dim'] = 1
    mtgnn_params['out_dim'] = 1
    mtgnn_params['xd'] = num_features
    return MTGNN(**mtgnn_params)


def _create_multigraph_gnn(num_nodes, seq_length, **kwargs):
    """Create MultigraphGNN model."""
    return MultigraphGNN(
        num_nodes=num_nodes,
        num_relations=kwargs.get('num_relations', 5),
        seq_length=seq_length + 1,
        in_dim=1,
        out_dim=1,
        residual_channels=kwargs.get('residual_channels', 64),
        conv_channels=kwargs.get('conv_channels', 64),
        skip_channels=kwargs.get('skip_channels', 64),
        end_channels=kwargs.get('end_channels', 128),
        layers=kwargs.get('layers', 4),
        kernel_set=kwargs.get('kernel_set', [1, 2]),
        kernel_size=kwargs.get('kernel_size', 2),
        dilation_exponential=kwargs.get('dilation_exponential', 2),
        dropout=kwargs.get('dropout', 0.5),
        layer_norm_affline=kwargs.get('layer_norm_affline', True),
        rgcn_num_bases=kwargs.get('rgcn_num_bases', None),
    )


def _create_lstm_model():
    input_size = 219
    hidden_size = 150
    output_size = 200
    external_forces_size = 19
    dense_output_size = 100
    return LSTMModel(input_size, hidden_size, output_size, external_forces_size, dense_output_size)



def model_forward(model, combined_input, model_type, config, device,
                   A_tilde=None, static_features=None,
                   edge_index=None, edge_type=None, edge_weight=None,
                   current_forces=None):
    """Model-agnostic forward pass dispatcher."""
    if model_type == 'MTGNN':
        if config['build_adj']:
            return model(combined_input, FE=static_features.to(device))
        return model(combined_input, A_tilde.to(device), FE=static_features.to(device))
    elif model_type == 'MultigraphGNN':
        return model(combined_input, edge_index, edge_type, edge_weight)
    elif model_type == 'LSTM':
        return model(combined_input, current_forces)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_val_rmse_per_node(model, eval_loader, device, future_window, W,
                              model_type, config, A_tilde, static_features,
                              num_piezo, edge_index, edge_type, edge_weight):
    """Compute per-node RMSE from validation batches (on scaled data)."""
    model.eval()
    node_squared_errors = np.zeros(num_piezo)
    node_counts = np.zeros(num_piezo)

    with torch.no_grad():
        for input_sequence, external_forces_sequence, target_sequence, mask_sequence in eval_loader:
            input_sequence = input_sequence.to(device)
            external_forces_sequence = external_forces_sequence.to(device)
            target_sequence = target_sequence.to(device)
            mask_sequence = mask_sequence.to(device)

            current_input = input_sequence
            with autocast(device_type="cuda"):
                predictions = []
                for t in range(future_window):
                    current_forces = external_forces_sequence[:, t:(W + t + 1), :]
                    combined_input = prepare_combined_input(current_input, current_forces)
                    output = model_forward(
                        model, combined_input, model_type, config, device,
                        A_tilde=A_tilde, static_features=static_features,
                        edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
                        current_forces=current_forces)
                    if model_type in ('MTGNN', 'MultigraphGNN'):
                        output = output[:, :, :num_piezo, 0]
                    predictions.append(output)
                    current_input = torch.cat((current_input[:, 1:, :], output), dim=1)
                predictions = torch.cat(predictions, dim=1)

            # Per-node squared error, respecting the missing-data mask
            sq_err = ((predictions - target_sequence) ** 2 * mask_sequence).cpu().numpy()
            mask_np = mask_sequence.cpu().numpy()
            # Sum across batch and time dimensions, per node
            node_squared_errors += sq_err.sum(axis=(0, 1))[:num_piezo]
            node_counts += mask_np.sum(axis=(0, 1))[:num_piezo]

    return np.sqrt(node_squared_errors / np.maximum(node_counts, 1))


def train(model, optimizer, loss_function, device, num_epochs, train_data, val_data,
          train_mask, val_mask, df_piezo_columns, num_piezo, static_features, A_tilde,
          F_w, W, config, model_type,
          edge_index=None, edge_type=None, edge_weight=None,
          run_dir=None):
    
    # Early stopping parameters
    early_stopping_patience = config.get('early_stopping_patience', 50)
    min_delta = config.get('min_delta', 0.001)
    best_loss = float('inf')

    # Node dropout state
    node_mask = None  # None = all nodes active; tensor of 1s/0s when dropout applied
    dropped_node_names = []

    # Learning rate scheduler setup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.get('scheduler_patience', 10))
    
    # Create output directories — only create legacy dirs when run_dir is not used
    if run_dir is None:
        os.makedirs(str(SAVED_MODELS_DIR), exist_ok=True)
        os.makedirs(str(TRAINING_RESULTS_DIR), exist_ok=True)
        os.makedirs(str(TRAINING_SUMMARIES), exist_ok=True)
        os.makedirs(str(RUN_PLOTS_AND_RESULTS), exist_ok=True)
        os.makedirs(str(SCATTER_PLOTS), exist_ok=True)
    os.makedirs("failed_runs", exist_ok=True)
    failed_runs_filepath = "failed_runs/failed_models.txt"

    losses_dict = {"train_losses": {}, "eval_losses": {}}
 
    best_model_filename = None
    
    scaler = GradScaler()

    for future_window in range(1, F_w + 1):  # Gradually increasing the future window
        start_time_window = time.time()  # Start time for the current window
        epoch_times = []  # List to store the duration of each epoch


        losses_dict["train_losses"][f"window_{future_window}"] = []
        losses_dict["eval_losses"][f"window_{future_window}"] = []
    
        # print(f"Training with future window: {future_window}")
        # Adjust the dataset for the current future window
    
        best_loss = float('inf')
        patience_counter = 0
    
        # Load the best model from the previous window if available
        if best_model_filename is not None:
            model.load_state_dict(torch.load(best_model_filename))
            # print(f"Loaded best model from {best_model_filename} for future window: {future_window}")
      
    
        # Model filename for the current future window
        if run_dir is not None:
            model_filename = os.path.join(str(run_dir), f"model_fw{future_window}.pt")
        else:
            model_filename = generate_model_filename(future_window=future_window, **config)
        # Remove the .pt extension from the model name for the losses
        model_name_for_losses = os.path.basename(model_filename).replace('.pt', '')


    
        # Check if the model already exists
        if os.path.exists(model_filename):
            model.load_state_dict(torch.load(model_filename))
            # print(f"Loaded model from {model_filename}. Skipping training.")
            continue
    
        # start_time_loading_train_dataset = time.time()

        batch_size = config.get('batch_size', 32)
        train_dataset = AutoregressiveTimeSeriesDataset(train_data, W, future_window, train_mask, num_piezo)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        eval_dataset = AutoregressiveTimeSeriesDataset(val_data, W, future_window, val_mask, num_piezo)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        # end_time_loading_train_dataset = time.time()
        # print(f"Loading training and eval dataset took {end_time_loading_train_dataset - start_time_loading_train_dataset:.2f} seconds")
            
        # restart early stopping for every window
        patience_counter = 0
        try:
        
            for epoch in range(num_epochs):
                start_time_epoch = time.time()  # Start time for the current epoch
    
                model.train()
                total_loss = 0
        
                for input_sequence, external_forces_sequence, target_sequence, mask_sequence in train_loader:
                    # Move data to the device
                    input_sequence = input_sequence.to(device)
                    external_forces_sequence = external_forces_sequence.to(device)
                    target_sequence = target_sequence.to(device)
                    mask_sequence = mask_sequence.to(device)
                    
                    optimizer.zero_grad()

                    with autocast(device_type='cuda'):
        
                      # Initialize autoregressive loop
                      current_input = input_sequence
                      predictions = []
          
                      
                      for t in range(future_window):
                          current_forces = external_forces_sequence[:, t : (W+t+1), :]
                          combined_input = prepare_combined_input(current_input, current_forces)

                          output = model_forward(
                              model, combined_input, model_type, config, device,
                              A_tilde=A_tilde, static_features=static_features,
                              edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
                              current_forces=current_forces)

                          if model_type in ('MTGNN', 'MultigraphGNN'):
                              output = output[:, :, :num_piezo, 0]
                          predictions.append(output)
                          next_input = output
                          current_input = torch.cat((current_input[:, 1:, :], next_input), dim=1)
          
                      # Multi-Step loss
                      predictions = torch.cat(predictions, dim=1)

                      # Apply node dropout mask (zero out dropped nodes' loss)
                      if node_mask is not None:
                          nm = node_mask.unsqueeze(0).unsqueeze(0)
                          predictions = predictions * nm
                          target_sequence = target_sequence * nm

                      predictions_masked = predictions * mask_sequence
                      target_masked = target_sequence * mask_sequence
                      loss = loss_function(predictions_masked, target_masked)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()
                    del input_sequence, external_forces_sequence, target_sequence, mask_sequence
            
                
        
                train_loss = total_loss / len(train_loader)
        
                model.eval()  # Set the model to evaluation mode
                total_eval_loss = 0
        
                with torch.no_grad():  # Disable gradient computations
                    for input_sequence, external_forces_sequence, target_sequence, mask_sequence in eval_loader:
                        # Move data to the device
                        input_sequence = input_sequence.to(device)
                        external_forces_sequence = external_forces_sequence.to(device)
                        target_sequence = target_sequence.to(device)
                        mask_sequence = mask_sequence.to(device)
        
                        current_input = input_sequence

                        with autocast(device_type = "cuda"):
                          predictions = []
                          for t in range(future_window):
                              current_forces = external_forces_sequence[:, t : (W+t + 1), :]
                              combined_input = prepare_combined_input(current_input, current_forces)

                              output = model_forward(
                                  model, combined_input, model_type, config, device,
                                  A_tilde=A_tilde, static_features=static_features,
                                  edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
                                  current_forces=current_forces)

                              if model_type in ('MTGNN', 'MultigraphGNN'):
                                  output = output[:, :, :num_piezo, 0]
                              predictions.append(output)

                              next_input = output
                              current_input = torch.cat((current_input[:, 1:, :], next_input), dim=1)
          
                          predictions = torch.cat(predictions, dim=1)

                          # Apply node dropout mask
                          if node_mask is not None:
                              nm = node_mask.unsqueeze(0).unsqueeze(0)
                              predictions = predictions * nm
                              target_sequence = target_sequence * nm

                          predictions_masked = predictions * mask_sequence
                          target_masked = target_sequence * mask_sequence
                          loss = loss_function(predictions_masked, target_masked)
                          total_eval_loss += loss.item()
          
                torch.cuda.empty_cache()  # Be cautious with frequent use
                
                eval_loss = total_eval_loss / len(eval_loader)
    
                losses_dict["train_losses"][f"window_{future_window}"].append({"epoch": epoch + 1, "loss": train_loss})
                losses_dict["eval_losses"][f"window_{future_window}"].append({"epoch": epoch + 1, "loss": eval_loss})
                
                # Print losses every 10 epochs
                if epoch % 10 == 9:
                    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.2e}, Eval Loss: {eval_loss:.2e}")
        
                # Dynamic node dropout — evaluate and mask after warmup epoch
                if (config.get('node_dropout') and node_mask is None
                        and epoch + 1 == config.get('node_dropout_warmup', 60)):
                    per_node_rmse = compute_val_rmse_per_node(
                        model, eval_loader, device, future_window, W,
                        model_type, config, A_tilde, static_features,
                        num_piezo, edge_index, edge_type, edge_weight)
                    mean_rmse = per_node_rmse.mean()
                    std_rmse = per_node_rmse.std()
                    threshold = mean_rmse + config.get('node_dropout_sd_threshold', 3.0) * std_rmse
                    dropped_indices = np.where(per_node_rmse > threshold)[0]

                    if len(dropped_indices) > 0:
                        node_mask = torch.ones(num_piezo, device=device)
                        node_mask[dropped_indices] = 0.0
                        dropped_node_names = [df_piezo_columns[i] for i in dropped_indices]

                        # Edge masking — zero out adjacency rows/cols for MTGNN
                        if A_tilde is not None:
                            full_mask = torch.ones(A_tilde.shape[0], device=A_tilde.device)
                            full_mask[:num_piezo] = node_mask.to(A_tilde.device)
                            A_tilde = A_tilde * full_mask.unsqueeze(0) * full_mask.unsqueeze(1)

                        # Edge masking — filter edges for MultigraphGNN
                        if edge_index is not None:
                            src_is_dropped = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
                            dst_is_dropped = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
                            for idx in dropped_indices:
                                src_is_dropped |= (edge_index[0] == idx)
                                dst_is_dropped |= (edge_index[1] == idx)
                            keep_edges = ~(src_is_dropped | dst_is_dropped)
                            edge_index = edge_index[:, keep_edges]
                            edge_type = edge_type[keep_edges]
                            if edge_weight is not None:
                                edge_weight = edge_weight[keep_edges]

                        print(f"Node dropout at epoch {epoch + 1}: {len(dropped_indices)} nodes dropped (threshold={threshold:.4f})")
                        print(f"  Dropped nodes: {dropped_node_names}")
                    else:
                        print(f"Node dropout at epoch {epoch + 1}: no outlier nodes found (threshold={threshold:.4f})")

                # Adaptive Learning Rate
                scheduler.step(eval_loss)

                # Early Stopping Check
                if eval_loss + min_delta < best_loss:
                    best_loss = eval_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), model_filename)
                    #print(f"Saved best model to {model_filename}")
                else:
                    patience_counter += 1
        
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
    
                end_time_epoch = time.time()  # End time for the current epoch
                epoch_times.append(end_time_epoch - start_time_epoch)  # Store the duration of the epoch

                # print(f"Epoch {epoch + 1} completed in {end_time_epoch - start_time_epoch:.2f} seconds")
            
            # # After completing training for the current window
            end_time_window = time.time()
            total_time_window = end_time_window - start_time_window  # Total time for the window
            average_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0  # Calculate the average time per epoch
            print(f"Average epoch time for future window {future_window}: {average_epoch_time:.2f} seconds")

            # print(f"Training for window {future_window} completed in {end_time_window - start_time_window:.2f} seconds")
                
        
            # Save losses JSON
            if run_dir is not None:
                loss_filename_filepath = os.path.join(str(run_dir), 'losses.json')
            else:
                loss_filename = f'losses_{model_name_for_losses}.json'
                os.makedirs(str(TRAINING_RESULTS_DIR), exist_ok=True)
                loss_filename_filepath = os.path.join(str(TRAINING_RESULTS_DIR), loss_filename)
            with open(loss_filename_filepath, 'w') as f:
                json.dump(losses_dict, f, indent=4)
                
                
            # At the end of training for each future window, update the best model filename
            best_model_filename = model_filename if os.path.exists(model_filename) else None
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Out of memory when processing {model_filename}.")
                with open(failed_runs_filepath, "a") as file:
                    file.write(f"{model_filename}\n")
                return dropped_node_names
            else:
                raise  # Re-raise the exception if it's not a memory error

    return dropped_node_names


def generate_configurations():
    """Generate configs: base config + Cartesian product of all parameter_variations."""
    base_config = define_base_configuration()
    params = list(parameter_variations.keys())
    variations = [parameter_variations[p] for p in params]

    if not params:
        return [base_config]

    configs = [base_config]
    for combo in itertools.product(*variations):
        cfg = base_config.copy()
        for p, v in zip(params, combo):
            cfg[p] = v
        configs.append(cfg)

    return configs

def main(run_all=True):

    summaries = []

    # Debug: confirm the loaded code and config
    print(f"[DEBUG] OUTPUTS_DIR = {OUTPUTS_DIR}")
    print(f"[DEBUG] parameter_variations = {parameter_variations}")

    configs = generate_configurations()
    base_config = configs[0]  # The first configuration is the base configuration
    total_runs = len(configs) if run_all else 1
    print(f"[DEBUG] Total configurations to run: {total_runs} (run_all={run_all})")



    for i, config in enumerate(configs[:total_runs], start=1):  # Limit the configs based on run_all flag
        differing_params = {k: v for k, v in config.items() if base_config.get(k) != v}
        differing_params_str = ', '.join([f'{key}: {value}' for key, value in differing_params.items()])
        
        if differing_params:
            print(f"Running configuration {i} of {total_runs} with variation: {differing_params_str}")
        else:
            print(f"Running base configuration {i} of {total_runs}")
        
        test_rmse_mean, test_rmse_std, geolayer_summary, dropped_node_names = run_training_and_evaluation(config)

        row = {
            "Timestamp":                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Model Type":               config["model_type"],
            "Graph Type":               config["graph_type"],
            "Percentage":               config["percentage"],
            "Piezometer Connections":   config["n_piezo_connected"],
            "Pump Connections":         config["n_pumps_connected"],
            'FIM':                      config["feature_importance_multiplier"],
            'Weight Mode':              config['weight_mode'],
            'Same Layer':               config['layer_constrain'],
            'Multiply_Exo_Weights':     config['multiply_exo_weights'],
            "W":                        config['W'],
            "F_w":                      config['F_w'],
            "Node Dropout":             config.get('node_dropout', False),
            "Dropped Nodes":            ", ".join(dropped_node_names) if dropped_node_names else "",
            "Directed Graph":           config.get('directed_graph', False),
            "Overall RMSE Mean":        test_rmse_mean,
            "Overall RMSE StdDev":      test_rmse_std,
        }

        for _, grp in geolayer_summary.iterrows():
            layer = grp["geolayer"]
            row[f"{layer} Mean"] = grp["RMSE Mean"]
            row[f"{layer} StdDev"] = grp["RMSE StdDev"]
          

        summaries.append(row)

        # Record the result
        record_result(config, test_rmse_mean, test_rmse_std)

    df_summary = pd.DataFrame(summaries)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(str(OUTPUTS_DIR), exist_ok=True)
    df_summary.to_csv(OUTPUTS_DIR / f"summary_{ts}.csv", index=False)
    print(f"→ Wrote run summary to {OUTPUTS_DIR / f'summary_{ts}.csv'}")

    # Append to persistent overall results table
    overall_path = OUTPUTS_DIR / "overall_results.csv"
    if overall_path.exists():
        df_existing = pd.read_csv(overall_path)
        df_combined = pd.concat([df_existing, df_summary], ignore_index=True)
    else:
        df_combined = df_summary
    df_combined.to_csv(overall_path, index=False)
    print(f"→ Updated overall results ({len(df_combined)} total rows) at {overall_path}")

    if run_all:
        analyze_results()


def run_training_and_evaluation(config):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Compute run output directory
    F_w = config.get('F_w', 3)
    model_base = os.path.basename(
        generate_model_filename(future_window=F_w, **config)
    ).replace('.pt', '')
    run_dir = OUTPUTS_DIR / config['graph_type'] / model_base
    os.makedirs(str(run_dir), exist_ok=True)
    print(f"Run output directory: {run_dir}")

    # Assuming process_data.main() prepares and returns the necessary datasets and GNN data
    train_data, val_data, test_data, train_mask, val_mask, test_mask, df_piezo_columns, pump_columns, locations_no_missing, scaler, mean_gw_elevation = process_data.main(config['synthetic_data'])
    train_data.to_csv(RANDOM_FOREST_TRAINING_DATA)
    A_tilde, static_features, pyg_graph = gnn_data_prep.main(df_piezo_columns, pump_columns, locations_no_missing, config['graph_type'], config['percentage'] , config['n_piezo_connected'], config['feature_importance_multiplier'], config['n_pumps_connected'], config['weight_mode'], config['layer_constrain'], config['ext_data'], config['multiply_exo_weights'], directed_graph=config.get('directed_graph', False), mean_gw_elevation=mean_gw_elevation)

    ahm = plot_adj_heatmap(A_tilde, output_dir=run_dir)
     
    # plot_sparsity_pattern(A_tilde, markersize=10)

    num_piezo = len(df_piezo_columns)
    num_features = static_features.shape[1]  # Assuming static_features is a tensor
    seq_length = config['W']  # sequence length
    num_nodes = A_tilde.shape[0]
    
    # Specify the sequence length (W) and future window size
    W = config['W']

    F_w = config.get('F_w', 3)
    model_type = config.get('model_type', 'MTGNN')
    model = create_model(
        num_features=num_features,
        num_nodes=num_nodes,
        seq_length=seq_length,
        **config
    ).to(device)

    # Prepare PyG graph tensors for GPU if needed
    if model_type == 'MultigraphGNN':
        edge_index = pyg_graph['edge_index'].to(device)
        edge_type = pyg_graph['edge_type'].to(device)
        edge_weight = pyg_graph['edge_weight'].to(device)
    else:
        edge_index = edge_type = edge_weight = None

    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    loss_function = nn.MSELoss()

    dropped_node_names = train(model, optimizer, loss_function, device, num_epochs=config.get('num_epochs', 200),
          train_data=train_data, val_data=val_data, train_mask=train_mask, val_mask=val_mask,
          df_piezo_columns=df_piezo_columns, num_piezo=num_piezo, static_features=static_features,
          A_tilde=A_tilde, F_w=F_w, W=W, config=config, model_type=model_type,
          edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,
          run_dir=run_dir)
    if dropped_node_names is None:
        dropped_node_names = []


    # Selecting first samples from training and testing datasets
    test_sample = AutoregressiveTimeSeriesDataset(test_data, input_window=W, max_future_window=100, missing_data_mask = test_mask, num_piezo = num_piezo)[1]
    test_input, test_predicted_model, test_target = make_predictions(
        model, test_sample, device, 100, W, A_tilde, static_features, num_piezo,
        modeltype=model_type, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight)
        
    # Transform predictions back to original scale

    test_predicted_model_ = inverse_transform_with_shape_adjustment(test_predicted_model.numpy(), scaler, num_piezo)
    test_input_ = inverse_transform_with_shape_adjustment(test_input.numpy(), scaler, num_piezo)
    test_target_ = inverse_transform_with_shape_adjustment(test_target.numpy(), scaler, num_piezo)

    test_rmse = calculate_rmse_per_piezometer(test_predicted_model_, test_target_, num_piezo)

    test_rmse_mean, test_rmse_std = print_mean_std(test_rmse, "Test RMSE")

    save_rmse_values(test_rmse, future_window=F_w, output_dir=run_dir, **config)

    # Plotting Model 1 Predictions
    _, _, _, mask_seq_test = test_sample
    start_date_test = test_data.index[0] 
    # plot_sequences(test_input_, test_predicted_model_, test_target_, df_piezo_columns, 'Model Evaluation', start_date_test, model_labels=('Prediction', '', ''), mask = mask_seq_test)
    color_dict_seq = plot_comparison_sequence(test_input_, test_predicted_model_, test_target_, start_date_test, df_piezo_columns, mask=mask_seq_test, selected_nodes=None, output_dir=run_dir)

    color_dict_dual = plot_comparison_sequence_dual_y(test_input_, test_predicted_model_, test_target_, start_date_test, mask_seq_test, test_rmse, df_piezo_columns, output_dir=run_dir)
    combined_color_dict = {**color_dict_seq, **color_dict_dual}

  

    # Load layer info
    layer_info = pd.read_csv(PIEZO_LAYER_INFORMATION).rename(columns=lambda x: x.strip())

    # Merge with RMSE values
    rmse_df = pd.DataFrame({
        'name': df_piezo_columns,
        'rmse': test_rmse
    })
    merged = rmse_df.merge(layer_info, on='name', how='left')

    # Build title from all relevant config options
    title_parts = [
        f"{model_type}",
        f"graph={config['graph_type']}",
        f"topo={config['n_piezo_connected']}",
        f"pumps={config['n_pumps_connected']}",
        f"W={config['W']}",
        f"weight_mode={config['weight_mode']}",
    ]
    if config.get('exclude_evap_precip'):
        title_parts.append(f"exclude_evap_precip={config['exclude_evap_precip']}")
    if config.get('perturb_weights'):
        title_parts.append("perturb_weights")
    if config.get('multiply_exo_weights'):
        title_parts.append("multiply_exo_weights")
    if config.get('directed_graph'):
        title_parts.append("directed")
    if config.get('node_dropout'):
        title_parts.append(f"node_dropout(warmup={config.get('node_dropout_warmup')}, sd={config.get('node_dropout_sd_threshold')})")
    title_str = " | ".join(title_parts)

    try:
      scatter = plot_rmse_3d_network(rmse_df, title_str)
      scatter.write_html(str(run_dir / 'rmse3d.html'), include_plotlyjs='cdn')
    except Exception as e:
      print(f"Skipped 3D RMSE plot for {title_str}: {e}")

    # Define summary function
    def summarize_by_column(col):
        grouped = merged.groupby(col)['rmse'].agg(['mean', 'std']).reset_index()
        grouped.columns = [col, 'RMSE Mean', 'RMSE StdDev']
        return grouped

    # Summarize by geolayer
    geolayer_summary = summarize_by_column('geolayer')

    # Summarize by regis_layer
    #regis_summary = summarize_by_column('regis_layer')

    # Print results
    print("\n📊 RMSE Summary by Geolayer:")
    print(geolayer_summary.to_string(index=False))

    return test_rmse_mean, test_rmse_std, geolayer_summary, dropped_node_names

if __name__ == "__main__":
    main(run_all=False) # for running only the base configuration

