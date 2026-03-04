import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
import json

import itertools
import sys

from data_preprocessing import process_data
from data_preprocessing import gnn_data_prep
from models.mtgnn import MTGNN
from models.lstm_model import LSTMModel
from data_preprocessing.dataset import AutoregressiveTimeSeriesDataset
from utils.training_utils import prepare_combined_input, make_predictions, inverse_transform_with_shape_adjustment, generate_model_filename, save_rmse_values, record_result, analyze_results, get_synthetic

from utils.metrics import calculate_rmse_per_piezometer, calculate_rmse_per_piezometer_moria, print_mean_std
from utils.visualization import plot_sequences, plot_sparsity_pattern, plot_comparison_sequence, plot_comparison_sequence_dual_y, plot_rmse_comparison, plot_rmse_3d_network, plot_adj_heatmap

from config import PIEZO_LAYER_INFORMATION, RANDOM_FOREST_TRAINING_DATA, SCATTER_PLOTS, TRAINING_SUMMARIES, SAVED_MODELS_DIR, TRAINING_RESULTS_DIR, RUN_PLOTS_AND_RESULTS, OUTPUTS_DIR

from train_config import define_base_configuration, parameter_variations, explicit_configs

import time
import datetime




from functools import partial


def create_model(num_features, num_nodes, seq_length, model_type, **kwargs):
    """Factory function to create the appropriate model based on model_type."""
    if model_type == 'MTGNN':
        return _create_mtgnn(num_features, num_nodes, seq_length, **kwargs)
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
        'layer_norm_affline', 'xd', 'multi_support'
    }}
    mtgnn_params['num_nodes'] = num_nodes
    mtgnn_params['seq_length'] = seq_length + 1
    mtgnn_params['in_dim'] = 1
    mtgnn_params['out_dim'] = 1
    mtgnn_params['xd'] = num_features
    return MTGNN(**mtgnn_params)


def _create_lstm_model():
    input_size = 219
    hidden_size = 150
    output_size = 200
    external_forces_size = 19
    dense_output_size = 100
    return LSTMModel(input_size, hidden_size, output_size, external_forces_size, dense_output_size)



def model_forward(model, combined_input, model_type, config, device,
                   A_tilde=None, static_features=None,
                   current_forces=None):
    """Model-agnostic forward pass dispatcher."""
    if model_type == 'MTGNN':
        if config.get('multi_support') or not config['build_adj']:
            return model(combined_input, A_tilde.to(device), FE=static_features.to(device))
        else:
            return model(combined_input, FE=static_features.to(device))
    elif model_type == 'LSTM':
        return model(combined_input, current_forces)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_val_rmse_per_node(model, eval_loader, device, future_window, W,
                              model_type, config, A_tilde, static_features,
                              num_piezo):
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
                        current_forces=current_forces)
                    if model_type == 'MTGNN':
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
          run_dir=None, eval_callback=None):
    
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
    
    scaler = GradScaler('cuda')

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
            if eval_callback is not None:
                eval_callback(future_window)
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
                              current_forces=current_forces)

                          if model_type == 'MTGNN':
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
                                  current_forces=current_forces)

                              if model_type == 'MTGNN':
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
        
                # Dynamic node dropout — evaluate and mask periodically after warmup
                dropout_warmup = config.get('node_dropout_warmup', 60)
                dropout_interval = config.get('node_dropout_check_interval', 20)
                dropout_eval_steps = config.get('node_dropout_eval_steps', 20)
                if (config.get('node_dropout')
                        and epoch + 1 >= dropout_warmup
                        and (epoch + 1 - dropout_warmup) % dropout_interval == 0):

                    # Build a multi-step evaluation loader to detect compounding errors
                    dropout_eval_dataset = AutoregressiveTimeSeriesDataset(
                        val_data, W, dropout_eval_steps, val_mask, num_piezo)
                    if len(dropout_eval_dataset) > 0:
                        dropout_eval_loader = DataLoader(
                            dropout_eval_dataset, batch_size=config.get('batch_size', 32), shuffle=False)

                        per_node_rmse = compute_val_rmse_per_node(
                            model, dropout_eval_loader, device, dropout_eval_steps, W,
                            model_type, config, A_tilde, static_features,
                            num_piezo)

                        # Only consider nodes that are still active
                        active_mask = np.ones(num_piezo, dtype=bool)
                        if node_mask is not None:
                            active_mask = node_mask.cpu().numpy().astype(bool)
                        active_rmse = per_node_rmse[active_mask]

                        mean_rmse = active_rmse.mean()
                        std_rmse = active_rmse.std()
                        sd_threshold = config.get('node_dropout_sd_threshold', 3.0)
                        threshold = mean_rmse + sd_threshold * std_rmse

                        # Diagnostics: show worst active nodes
                        active_indices = np.where(active_mask)[0]
                        worst_active = active_indices[np.argsort(per_node_rmse[active_mask])[-5:][::-1]]
                        print(f"[Node Dropout] epoch={epoch+1}, eval_steps={dropout_eval_steps}, "
                              f"mean_rmse={mean_rmse:.4f}, std={std_rmse:.4f}, "
                              f"threshold={threshold:.4f} (mean + {sd_threshold}*SD)")
                        for wi in worst_active:
                            sds_above = (per_node_rmse[wi] - mean_rmse) / std_rmse if std_rmse > 0 else 0
                            status = "ACTIVE" if active_mask[wi] else "DROPPED"
                            print(f"  {df_piezo_columns[wi]}: RMSE={per_node_rmse[wi]:.4f} ({sds_above:.1f} SD above mean) [{status}]")

                        # Find new nodes to drop (among still-active nodes)
                        new_drops = np.where(active_mask & (per_node_rmse > threshold))[0]

                        if len(new_drops) > 0:
                            if node_mask is None:
                                node_mask = torch.ones(num_piezo, device=device)
                            node_mask[new_drops] = 0.0
                            new_drop_names = [df_piezo_columns[i] for i in new_drops]
                            dropped_node_names.extend(new_drop_names)

                            # Edge masking — zero out adjacency rows/cols for MTGNN
                            if A_tilde is not None:
                                full_mask = torch.ones(A_tilde.shape[0], device=A_tilde.device)
                                full_mask[:num_piezo] = node_mask.to(A_tilde.device)
                                A_tilde = A_tilde * full_mask.unsqueeze(0) * full_mask.unsqueeze(1)

                            print(f"Node dropout at epoch {epoch + 1}: {len(new_drops)} new nodes dropped (threshold={threshold:.4f})")
                            print(f"  Newly dropped: {new_drop_names}")
                            print(f"  All dropped nodes: {dropped_node_names}")
                        else:
                            print(f"Node dropout at epoch {epoch + 1}: no new outlier nodes (threshold={threshold:.4f})")

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

                # Suppress early stopping until after first dropout check so it gets a chance to run
                es_min_epoch = config.get('node_dropout_warmup', 60) if config.get('node_dropout') else 0
                if patience_counter >= early_stopping_patience and epoch + 1 > es_min_epoch:
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

            # Evaluate and produce output after each F_w training step
            if eval_callback is not None:
                eval_callback(future_window)
                
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
    """Generate configs from explicit_configs (if non-empty) or Cartesian product of parameter_variations."""
    base_config = define_base_configuration()

    # Explicit config list takes priority over Cartesian product
    if explicit_configs:
        configs = []
        for overrides in explicit_configs:
            cfg = base_config.copy()
            cfg.update(overrides)
            configs.append(cfg)
        return configs

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
        
        step_results, dropped_node_names = run_training_and_evaluation(config)

        for fw_step, test_rmse_mean, test_rmse_std, geolayer_summary in step_results:
            row = {
                "Timestamp":                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Model Type":               config["model_type"],
                "Graph Type":               config["graph_type"],
                "Percentage":               config["percentage"],
                "Piezometer Connections":   config["n_piezo_connected"],
                "Pump Connections":         config["n_pumps_connected"],
                'FIM':                      config["feature_importance_multiplier"],
                'Weight Mode':              config['weight_mode'],
                'RF Weight Range':          f"{config.get('rf_weight_min', '')}-{config.get('rf_weight_max', '')}" if config['weight_mode'] in ('variable', 'full') else "",
                'VIM Min':                  config.get('rf_vim_min', '') if config['weight_mode'] == 'full' else "",
                'Multi-Support':            config.get('multi_support', False),
                'Adaptive Init':            config.get('adaptive_graph_type', '') if config.get('multi_support') else "",
                'Same Layer':               config['layer_constrain'],
                "W":                        config['W'],
                "F_w (trained)":            fw_step,
                "F_w (config)":             config['F_w'],
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

            # Save results incrementally after each run (preserves existing formatting)
            os.makedirs(str(OUTPUTS_DIR), exist_ok=True)
            overall_path = OUTPUTS_DIR / "overall_results.xlsx"
            df_row = pd.DataFrame([row])
            if overall_path.exists():
                try:
                    from openpyxl import load_workbook
                    wb = load_workbook(str(overall_path))
                    ws = wb.active
                    for r in df_row.itertuples(index=False):
                        ws.append(list(r))
                    wb.save(str(overall_path))
                except Exception as e:
                    print(f"⚠ Could not append to {overall_path} ({e}). Backing up and creating new file.")
                    backup = overall_path.with_suffix('.xlsx.bak')
                    overall_path.rename(backup)
                    df_row.to_excel(overall_path, index=False)
            else:
                df_row.to_excel(overall_path, index=False)
            print(f"→ Appended F_w={fw_step} result for run {i} to {overall_path}")

    # Also write a timestamped summary of this session
    df_summary = pd.DataFrame(summaries)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_summary.to_csv(OUTPUTS_DIR / f"summary_{ts}.csv", index=False)
    print(f"→ Wrote session summary to {OUTPUTS_DIR / f'summary_{ts}.csv'}")

    if run_all:
        analyze_results()


def evaluate_and_output(model, config, test_data, test_mask, df_piezo_columns, num_piezo,
                        scaler, A_tilde, static_features, W, device, run_dir, model_type,
                        dropped_node_names, fw_step):
    """Run 100-step autoregressive test evaluation and produce all outputs for a given F_w training step."""

    # Create F_w-specific output subdirectory
    step_dir = run_dir / f"eval_fw{fw_step}"
    os.makedirs(str(step_dir), exist_ok=True)

    test_sample = AutoregressiveTimeSeriesDataset(test_data, input_window=W, max_future_window=100, missing_data_mask=test_mask, num_piezo=num_piezo)[1]
    test_input, test_predicted_model, test_target = make_predictions(
        model, test_sample, device, 100, W, A_tilde, static_features, num_piezo,
        modeltype=model_type)

    test_predicted_model_ = inverse_transform_with_shape_adjustment(test_predicted_model.numpy(), scaler, num_piezo)
    test_input_ = inverse_transform_with_shape_adjustment(test_input.numpy(), scaler, num_piezo)
    test_target_ = inverse_transform_with_shape_adjustment(test_target.numpy(), scaler, num_piezo)

    test_rmse = calculate_rmse_per_piezometer(test_predicted_model_, test_target_, num_piezo)
    test_rmse_mean, test_rmse_std = print_mean_std(test_rmse, f"Test RMSE (after F_w={fw_step} training)")

    save_rmse_values(test_rmse, future_window=fw_step, output_dir=step_dir, **config)

    # Plotting
    _, _, _, mask_seq_test = test_sample
    start_date_test = test_data.index[0]
    plot_freq = config.get('resampling_freq', 'W')
    if plot_freq is None:
        plot_freq = pd.infer_freq(test_data.index) or '3h'

    color_dict_seq = plot_comparison_sequence(test_input_, test_predicted_model_, test_target_, start_date_test, df_piezo_columns, mask=mask_seq_test, selected_nodes=None, output_dir=step_dir, freq=plot_freq)
    color_dict_dual = plot_comparison_sequence_dual_y(test_input_, test_predicted_model_, test_target_, start_date_test, mask_seq_test, test_rmse, df_piezo_columns, output_dir=step_dir, freq=plot_freq)

    # Layer info and RMSE summary
    layer_info = pd.read_csv(PIEZO_LAYER_INFORMATION).rename(columns=lambda x: x.strip())
    rmse_df = pd.DataFrame({'name': df_piezo_columns, 'rmse': test_rmse})
    if dropped_node_names:
        rmse_df = rmse_df[~rmse_df['name'].isin(dropped_node_names)].reset_index(drop=True)
    merged = rmse_df.merge(layer_info, on='name', how='left')

    # Build title
    title_parts = [
        f"{model_type}",
        f"graph={config['graph_type']}",
        f"topo={config['n_piezo_connected']}",
        f"pumps={config['n_pumps_connected']}",
        f"W={config['W']}",
        f"F_w={fw_step}",
        f"weight_mode={config['weight_mode']}",
    ]
    if config.get('exclude_evap_precip'):
        title_parts.append(f"exclude_evap_precip={config['exclude_evap_precip']}")
    if config.get('perturb_weights'):
        title_parts.append("perturb_weights")
    if config.get('directed_graph'):
        title_parts.append("directed")
    if config.get('node_dropout'):
        title_parts.append(f"node_dropout(warmup={config.get('node_dropout_warmup')}, sd={config.get('node_dropout_sd_threshold')})")
    title_str = " | ".join(title_parts)

    try:
        scatter = plot_rmse_3d_network(rmse_df, title_str, adj_matrix=A_tilde)
        scatter.write_html(str(step_dir / 'rmse3d.html'), include_plotlyjs='cdn')
    except Exception as e:
        print(f"Skipped 3D RMSE plot for {title_str}: {e}")

    geolayer_summary = merged.groupby('geolayer')['rmse'].agg(['mean', 'std']).reset_index()
    geolayer_summary.columns = ['geolayer', 'RMSE Mean', 'RMSE StdDev']

    print(f"\n📊 RMSE Summary by Geolayer (F_w={fw_step}):")
    print(geolayer_summary.to_string(index=False))

    return test_rmse_mean, test_rmse_std, geolayer_summary


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
    train_data, val_data, test_data, train_mask, val_mask, test_mask, df_piezo_columns, pump_columns, locations_no_missing, scaler, mean_gw_elevation = process_data.main(
        config['synthetic_data'],
        resampling_freq=config.get('resampling_freq', 'W')
    )
    train_data.to_csv(RANDOM_FOREST_TRAINING_DATA)
    A_tilde, static_features = gnn_data_prep.main(df_piezo_columns, pump_columns, locations_no_missing, config['graph_type'], config['percentage'] , config['n_piezo_connected'], config['feature_importance_multiplier'], config['n_pumps_connected'], config['weight_mode'], config['layer_constrain'], directed_graph=config.get('directed_graph', False), mean_gw_elevation=mean_gw_elevation, rf_weight_min=config.get('rf_weight_min', 0.08), rf_weight_max=config.get('rf_weight_max', 0.2), rf_vim_min=config.get('rf_vim_min', 0.01))

    heatmap_title = (f"{config.get('model_type', 'MTGNN')} | graph={config['graph_type']} | "
                     f"weight_mode={config['weight_mode']}<br>"
                     f"piezo={config['n_piezo_connected']} | pumps={config['n_pumps_connected']} | "
                     f"W={config['W']}")
    if config.get('node_dropout'):
        heatmap_title += f" | node_dropout(warmup={config.get('node_dropout_warmup')})"
    if config.get('directed_graph'):
        heatmap_title += " | directed"
    ahm = plot_adj_heatmap(A_tilde, output_dir=run_dir, title=heatmap_title)

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

    # Multi-support: compute adaptive adjacency init and seed the model
    if config.get('multi_support') and model_type == 'MTGNN':
        adap_gt = config.get('adaptive_graph_type', 'rf')
        adap_wm = config.get('adaptive_weight_mode', 'fixed')
        print(f"Multi-support: computing adaptive init adjacency "
              f"(graph_type={adap_gt}, weight_mode={adap_wm})")
        adaptive_adj, _ = gnn_data_prep.main(
            df_piezo_columns, pump_columns, locations_no_missing,
            adap_gt, config['percentage'], config['n_piezo_connected'],
            config['feature_importance_multiplier'], config['n_pumps_connected'],
            adap_wm, config['layer_constrain'],
            directed_graph=config.get('directed_graph', False),
            mean_gw_elevation=mean_gw_elevation,
            rf_weight_min=config.get('rf_weight_min', 0.08),
            rf_weight_max=config.get('rf_weight_max', 0.2),
            rf_vim_min=config.get('rf_vim_min', 0.01))
        model.init_adaptive_adj(adaptive_adj)
        print(f"Adaptive adjacency initialized from {adap_gt} "
              f"(shape={adaptive_adj.shape}, "
              f"nonzero={np.count_nonzero(adaptive_adj)})")

    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    loss_function = nn.MSELoss()

    # Collect results for each F_w step
    step_results = []

    dropped_node_names = train(model, optimizer, loss_function, device, num_epochs=config.get('num_epochs', 200),
          train_data=train_data, val_data=val_data, train_mask=train_mask, val_mask=val_mask,
          df_piezo_columns=df_piezo_columns, num_piezo=num_piezo, static_features=static_features,
          A_tilde=A_tilde, F_w=F_w, W=W, config=config, model_type=model_type,
          run_dir=run_dir,
          eval_callback=lambda fw_step: step_results.append(
              (fw_step, *evaluate_and_output(
                  model, config, test_data, test_mask, df_piezo_columns, num_piezo,
                  scaler, A_tilde, static_features, W, device, run_dir, model_type,
                  [],
                  fw_step))
          ))
    if dropped_node_names is None:
        dropped_node_names = []

    # If train() didn't trigger callbacks (e.g. all steps loaded from cache),
    # evaluate once at the final F_w
    if not step_results:
        rmse_mean, rmse_std, geo_summary = evaluate_and_output(
            model, config, test_data, test_mask, df_piezo_columns, num_piezo,
            scaler, A_tilde, static_features, W, device, run_dir, model_type,
            dropped_node_names, F_w)
        step_results.append((F_w, rmse_mean, rmse_std, geo_summary))

    return step_results, dropped_node_names

if __name__ == "__main__":
    main(run_all=False) # for running only the base configuration

