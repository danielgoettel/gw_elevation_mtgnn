import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from pathlib import Path
from config import ADJ_MATRIX_PATH, PREPROCESSED_DIR

def plot_two_random_columns(df, one_month_only = True):
    # Ensure the DataFrame has at least 2 columns
    if df.shape[1] < 2:
        print("The DataFrame has fewer than 2 columns!")
        return

    # Randomly select two distinct columns
    random_columns = random.sample(list(df.columns), 2)

    # Slice the DataFrame for one week if one_week_only is True
    if one_month_only:
        # Assuming the DataFrame is sorted by date
        start_date = df.index.min()
        end_date = start_date + pd.Timedelta(days=30)
        df = df.loc[start_date:end_date]
    plt.figure(figsize=(10, 6))
    
    # Plot the first random column
    plt.plot(df.index, df[random_columns[0]], label=random_columns[0], color='blue')
    
    # Plot the second random column
    plt.plot(df.index, df[random_columns[1]], label=random_columns[1], color='red')
    
    plt.title("Two Random Columns from the DataFrame")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_distributions(original_data, normalized_data, df_name):
    plt.figure(figsize=(12, 5))

    # Plot original distribution
    plt.subplot(1, 2, 1)
    plt.hist(original_data, bins=50, alpha=0.7)
    plt.title(f'Original Distribution of {df_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot normalized distribution
    plt.subplot(1, 2, 2)
    plt.hist(normalized_data, bins=50, alpha=0.7, color='orange')
    plt.title(f'Normalized Distribution of {df_name}')
    plt.xlabel('Normalized Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()



def plot_sparsity_pattern(matrix, markersize=5, figsize=(10, 10)):
    """
    Plots the sparsity pattern of a matrix.
    
    Parameters:
    - matrix: A PyTorch tensor representing the matrix whose sparsity pattern is to be plotted.
    - markersize: Size of the markers for non-zero elements.
    - figsize: Size of the figure.
    """
    # Convert the PyTorch tensor to a NumPy array
    matrix_np = matrix.numpy()
    
    # Create a figure to plot
    plt.figure(figsize=figsize)
    # Use spy to visualize the sparsity pattern, specifying black color for dots
    plt.spy(matrix_np, markersize=markersize, markeredgecolor='black', markerfacecolor='black')
    
    # Remove title and labels
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    # Set larger numbers in the axis
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Show plot
    plt.show()


def plot_losses(train_losses, eval_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.show()
    


def apply_mask(data, mask):
    # Convert the mask to a NumPy array if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    # Set data to NaN where mask is 0
    masked_data = np.where(mask == 0, np.nan, data)
    return masked_data


def plot_sequences(input_seq, predicted_seq_model1, target_seq, df_piezo_columns, title_prefix, start_date, predicted_seq_model2=None, predicted_seq_model3=None, model_labels=('Prediction Model 1', 'Prediction Model 2', 'Prediction Model 3'), mask=None, selected_nodes=None):
    num_time_steps = input_seq.shape[0] + predicted_seq_model1.shape[0]
    dates = pd.date_range(start=start_date, periods=num_time_steps, freq='W')

    if mask is not None:
        predicted_seq_model1 = apply_mask(predicted_seq_model1, mask)
        target_seq = apply_mask(target_seq, mask)

    # If specific nodes are not provided, randomly sample three nodes
    if selected_nodes is None:
        selected_nodes = random.sample(range(input_seq.shape[1]), 3)

    input_data = input_seq[:, selected_nodes].numpy() if torch.is_tensor(input_seq) else input_seq[:, selected_nodes]
    predicted_data_model1 = predicted_seq_model1[:, selected_nodes].numpy() if torch.is_tensor(predicted_seq_model1) else predicted_seq_model1[:, selected_nodes]
    target_data = target_seq[:, selected_nodes].numpy() if torch.is_tensor(target_seq) else target_seq[:, selected_nodes]

    full_predicted_sequence_model1 = np.concatenate((input_data, predicted_data_model1), axis=0)
    full_target_sequence = np.concatenate((input_data, target_data), axis=0)

    for i, node in enumerate(selected_nodes):
        plt.figure(figsize=(10, 4))
        node_name = df_piezo_columns[node]
        # plt.title(f'{title_prefix} - Piezometer {node_name}')
        plt.plot(dates, full_predicted_sequence_model1[:, i], label=model_labels[0], color='red', linestyle='--')

        if predicted_seq_model2 is not None:
            predicted_data_model2 = predicted_seq_model2[:, selected_nodes].numpy() if torch.is_tensor(predicted_seq_model2) else predicted_seq_model2[:, selected_nodes]
            full_predicted_sequence_model2 = np.concatenate((input_data, predicted_data_model2), axis=0)
            plt.plot(dates, full_predicted_sequence_model2[:, i], label=model_labels[1], color='green', linestyle='--')

        if predicted_seq_model3 is not None:
            predicted_data_model3 = predicted_seq_model3[:, selected_nodes].numpy() if torch.is_tensor(predicted_seq_model3) else predicted_seq_model3[:, selected_nodes]
            full_predicted_sequence_model3 = np.concatenate((input_data, predicted_data_model3), axis=0)
            plt.plot(dates, full_predicted_sequence_model3[:, i], label=model_labels[2], color='purple', linestyle='--')

        plt.plot(dates, full_target_sequence[:, i], label='Target', color='blue')
        plt.axvline(x=dates[input_seq.shape[0]], color='gray', linestyle='--', label='Prediction Start')
        plt.ylabel('Groundwater level [cm NAP]')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()
        
from itertools import cycle

def plot_comparison_sequence(input_seq, predicted_seq, target_seq, start_date, train_piezo_columns, mask=None, selected_nodes=None):
    num_time_steps = input_seq.shape[0] + predicted_seq.shape[0]
    dates = pd.date_range(start=start_date, periods=num_time_steps, freq='W')

    if mask is not None:
        # predicted_seq = apply_mask(predicted_seq, mask)
        target_seq = apply_mask(target_seq, mask)
        
    if selected_nodes is None:
        num_nodes = input_seq.shape[1]  # Get the number of nodes from the input sequence
        selected_nodes = random.sample(range(num_nodes), min(4, num_nodes))  # Ensure we don't sample more nodes than available

    input_data = input_seq[:, selected_nodes].numpy() if torch.is_tensor(input_seq) else input_seq[:, selected_nodes]
    predicted_data = predicted_seq[:, selected_nodes].numpy() if torch.is_tensor(predicted_seq) else predicted_seq[:, selected_nodes]
    target_data = target_seq[:, selected_nodes].numpy() if torch.is_tensor(target_seq) else target_seq[:, selected_nodes]

    full_predicted_sequence = np.concatenate((input_data, predicted_data), axis=0)
    full_target_sequence = np.concatenate((input_data, target_data), axis=0)

    plt.figure(figsize=(10, 4))


    # # Define a custom color cycle that excludes red, green, and black
    # colors = plt.cm.tab10.colors  # Default matplotlib color cycle
    # colors = [color for i, color in enumerate(colors) if i not in {0, 2, 3}]  # Exclude red (0), green (2), and black-like (3)
    # color_cycle = cycle(colors)
    
    # Get the 'tab10' color cycle, which is a default color cycle in matplotlib
    colors = plt.get_cmap('tab10').colors
    
    # Exclude specific colors by their index in the 'tab10' cycle
    # In 'tab10', index 0 is blue, index 1 is orange, index 2 is green, index 3 is red, etc.
    # If you want to exclude red (3), green (2), and any color that looks like black, we focus on excluding 3 and 2
    # Note: 'tab10' does not include a black color, but we will adhere to the exclusion criteria provided
    colors = [color for i, color in enumerate(colors) if i not in {2, 3}]  # Excludes green and red
    
    # Create a cycle of the remaining colors
    color_cycle = cycle(colors)


    color_dict = {}  # Dictionary to store node colors
    for i, node in enumerate(selected_nodes):
        # color = f'C{i+2}'  # Start from C2 to avoid red and green
        color = next(color_cycle)  # Get the next color from the cycle

        color_dict[train_piezo_columns[node]] = color
        
        # color_dict[f'Node {node+1}'] = color  # Store the color with the node label
        plt.plot(dates, full_predicted_sequence[:, i], label=f'Prediction (Node {node+1})', color=color, linestyle='--')
        plt.plot(dates, full_target_sequence[:, i], color=color)

    plt.axvline(x=dates[input_seq.shape[0]], color='gray', linestyle='--', label='Prediction Start')
    plt.ylabel('Groundwater level [cm NAP]', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.xticks(rotation=45)

    # Create custom legends and position it at the bottom right
    # legend_elements = [plt.Line2D([0], [0], color='k', linestyle='-', label='Target'),
    #                    plt.Line2D([0], [0], color='k', linestyle='--', label='Prediction')]
    # plt.legend(handles=legend_elements, loc='lower right', fontsize=12)


    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Prediction Start'),
        plt.Line2D([0], [0], color='k', linestyle='-', label='Target'),
        plt.Line2D([0], [0], color='k', linestyle='--', label='Prediction'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)


    # Save the figure as a PDF in the current working directory
    figure_directory = '.'  # Current directory
    file_name = 'comparison_sequence.pdf'
    plt.savefig(f'{figure_directory}/{file_name}', format='pdf', bbox_inches='tight')


    plt.show()
    return color_dict  


def plot_comparison_sequence_dual_y(input_seq, predicted_seq, target_seq, start_date, mask, test_rmse, train_piezo_columns ):
    
    # Find the best (lowest) RMSE values
    best_rmse_values = np.sort(test_rmse)[:3]
    
    # Find the worst (highest) RMSE values
    worst_rmse_values = np.sort(test_rmse)[-3:]
    
    print("Best RMSE values:", best_rmse_values)
    print("Worst RMSE values:", worst_rmse_values)
    
    best_rmse_indices = np.argsort(test_rmse)[:3]
    worst_rmse_indices = np.argsort(test_rmse)[-3:]
    
    
    # Get the names of the best performing piezometers

    best_rmse_piezometers = [train_piezo_columns[index] for index in best_rmse_indices]

    print("Best performing piezometers based on RMSE:", best_rmse_piezometers)
    
    # Get the names of the worst performing piezometers
    worst_rmse_piezometers = [train_piezo_columns[index] for index in worst_rmse_indices]

    print("Worst performing piezometers based on RMSE:", worst_rmse_piezometers)

    best_piezometer = best_rmse_piezometers[0]  # The very best
    worst_piezometer = worst_rmse_piezometers[1]  # The very worst would be [2]




    num_time_steps = input_seq.shape[0] + predicted_seq.shape[0]
    dates = pd.date_range(start=start_date, periods=num_time_steps, freq='W')

    if mask is not None:
        # predicted_seq = apply_mask(predicted_seq, mask)
        target_seq = apply_mask(target_seq, mask)
        

    selected_nodes = []

    # Map piezometer names to indices
    if best_piezometer is not None:
        selected_nodes.append(train_piezo_columns.index(best_piezometer))

    if worst_piezometer is not None:
        selected_nodes.append(train_piezo_columns.index(worst_piezometer))

    input_data = input_seq[:, selected_nodes].numpy() if torch.is_tensor(input_seq) else input_seq[:, selected_nodes]
    predicted_data = predicted_seq[:, selected_nodes].numpy() if torch.is_tensor(predicted_seq) else predicted_seq[:, selected_nodes]
    target_data = target_seq[:, selected_nodes].numpy() if torch.is_tensor(target_seq) else target_seq[:, selected_nodes]

    full_predicted_sequence = np.concatenate((input_data, predicted_data), axis=0)
    full_target_sequence = np.concatenate((input_data, target_data), axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color_best = 'green'
    color_worst = 'red'
    
    # Plotting the best piezometer on ax1
    if best_piezometer is not None:
        ax1.plot(dates, full_predicted_sequence[:, 0], label=f'Best Prediction', color=color_best, linestyle='--')
        ax1.plot(dates, full_target_sequence[:, 0], label=f'Target', color=color_best)
        ax1.set_ylabel('Groundwater Level [cm NAP]', fontsize=14, color='black')
        ax1.tick_params(axis='y', labelcolor=color_best)

    # Plotting the worst piezometer on ax2
    if worst_piezometer is not None:
        ax2.plot(dates, full_predicted_sequence[:, -1], label=f'Worst Prediction', color=color_worst, linestyle='--')
        ax2.plot(dates, full_target_sequence[:, -1], label=f'Target', color=color_worst)
        ax2.set_ylabel('', fontsize=14, color=color_worst)
        ax2.tick_params(axis='y', labelcolor=color_worst)

    ax1.axvline(x=dates[input_seq.shape[0]], color='gray', linestyle='--', label='Prediction Start')
    ax1.set_xlabel('Date', fontsize=14)
    plt.xticks(rotation=45)

    fig.tight_layout()  # Adjust layout to make room for the legend

    # Create custom legends and position it at the right
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=12)

    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Prediction Start'),
        plt.Line2D([0], [0], color='k', linestyle='-', label='Target'),
        plt.Line2D([0], [0], color='k', linestyle='--', label='Prediction'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)

    # Save the figure as a PDF in the current working directory
    figure_directory = '.'  # Current directory
    file_name = 'comparison_sequence_best_worst.pdf'
    plt.savefig(f'{figure_directory}/{file_name}', format='pdf', bbox_inches='tight')

    plt.show()
    
    # Save the colors
    color_dict = {} 
    color_dict[best_piezometer] = color_best
    color_dict[worst_piezometer] = color_worst
    return color_dict  



def plot_rmse_comparison(model1_rmse, model2_rmse, label1, label2, color_dict):
    """
    Generates a scatter plot comparing RMSE values from two models.

    Parameters:
    model1_rmse (dict): Dictionary of RMSE values for Model 1 with piezometer names as keys.
    model2_rmse (dict): Dictionary of RMSE values for Model 2 with piezometer names as keys.
    label1 (str): Label for Model 1 to be displayed in the plot.
    label2 (str): Label for Model 2 to be displayed in the plot.
    file_name (str): The name of the file to save the plot.
    """
    
    plt.figure(figsize=(10, 8))  # Larger figure size

    # Extracting RMSE values and ensuring the keys match
    keys = model1_rmse.keys()
    values1 = [model1_rmse[key] for key in keys]
    values2 = [model2_rmse[key] for key in keys]

    # Scatter plot
    # plt.scatter(values1, values2, alpha=0.7, label="RMSE Comparison")
    # Scatter plot using colors from the color_dict
    for key in keys:
        if key in color_dict:
            # Special nodes get a larger marker size
            plt.scatter(model1_rmse[key], model2_rmse[key], alpha=0.7, 
                        color=color_dict.get(key, 'black'), s=200,marker='*')  # Larger size for special nodes
                    # edgecolors='gold', linewidths=5)  # Option 1

        else:
            # Regular nodes get the default marker size
            plt.scatter(model1_rmse[key], model2_rmse[key], alpha=0.7, 
                        color=color_dict.get(key, 'black'), s=50)  # Default size for regular nodes

    
    # Max value for x and y axis
    max_value = max(max(values1), max(values2))
    
    # Plotting the y=x line
    plt.plot([0, max_value], [0, max_value], 'r--', label=f'{label1} = {label2}')
    
    # Labels and title with larger font size
    plt.xlabel(f'{label1} RMSE', fontsize=14)
    plt.ylabel(f'{label2} RMSE', fontsize=14)
    # plt.title('RMSE Comparison between Two Models', fontsize=16)
    
    # Enlarging tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Legend
    plt.legend(fontsize=12)
    
    # Grid
    plt.grid(True)
    
    # Save the figure
    figure_directory = '.'  # Current directory
    file_name = 'scatterplot_RMSE.pdf'
    plt.savefig(f'{figure_directory}/{file_name}', format='pdf', bbox_inches='tight')
    
    # Show the plot
    plt.show()



def plot_rmse_3d_network(
    rmse_df: pd.DataFrame,
    plot_title = None, 
    color_columns = ['Type','geolayer','regis_layer'],
    size_range   = (5, 25),
    include_non_piezometers: bool = False
):
    # 1) Load adjacency
    raw_A = torch.load(ADJ_MATRIX_PATH, weights_only=False)
    A = raw_A.cpu().numpy() if isinstance(raw_A, torch.Tensor) else np.array(raw_A)

    # 2) Load nodes & merge RMSE
    nodes = pd.read_csv(Path(PREPROCESSED_DIR)/"nodes_complete.csv")
    if not include_non_piezometers:
        nodes = nodes[nodes['Type']=='Piezometer'].reset_index(drop=True)
    # left merge preserves all nodes; rmse NaN for non-piezometers
    df = nodes.merge(rmse_df, on='name', how='left')
    # set missing RMSE to min observed so they get smallest marker size
    min_rmse = rmse_df['rmse'].min()
    df['rmse'] = df['rmse'].fillna(min_rmse)
    df = df.reset_index(drop=True)

    coords = df[['x','y','z']].values
    names  = df['name'].tolist()
    rmses  = df['rmse'].values
    

    # 3) Marker sizes from RMSE
    rmin, rmax = rmses.min(), rmses.max()
    smin, smax = size_range
    sizes = smin + (rmses)/(150)*(smax - smin)

    # 4) Map original node index → position in nodes.csv
    orig_names = pd.read_csv(Path(PREPROCESSED_DIR)/"nodes.csv")['name'].tolist()
    idx_map = {n:i for i,n in enumerate(orig_names)}
    filtered_idx = [idx_map[n] for n in names]

    N = len(filtered_idx)

        # 5) Precompute raw edges and their weights
    edge_data = []
    for u in range(N):
        i = filtered_idx[u]
        for v in range(u+1, N):
            j = filtered_idx[v]
            w = A[i, j]
            if w > 0:
                x0,y0,z0 = coords[u]
                x1,y1,z1 = coords[v]
                edge_data.append({
                    "ends": ((x0,y0,z0), (x1,y1,z1)),
                    "weight": w,
                    # pick one “owner” group for legend toggling — 
                    # here we inherit from the u‐node’s first color_column
                    "group": f"{color_columns[0]} = {df.iloc[u][color_columns[0]]}"
                })

    # scale widths so the smallest weight → width = 1
    #w_min = min(e["weight"] for e in edge_data)
    for e in edge_data:
        e["width"] = e["weight"] #/ w_min

    # 6) build one Scatter3d per edge
    traces = []
    for e in edge_data:
        (x0,y0,z0), (x1,y1,z1) = e["ends"]
        traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(width=e["width"] * 3, color='black'),
            hoverinfo='text',
            text=[f"Weight: {e['weight']:.3f}"],
            legendgroup=e["group"],   # tie it to the node‐trace group
            showlegend=False          # only the node trace shows in the legend
        ))

    #append node traces
    for col in color_columns:
        for val in sorted(df[col].unique()):
            mask = df[col] == val
            trace_name = f"{col} = {val}"
            traces.append(go.Scatter3d(
                x=coords[mask,0], y=coords[mask,1], z=coords[mask,2],
                mode='markers',
                hoverinfo='text',
                hovertext=df['name'][mask] + "<br>RMSE: " + df['rmse'][mask].round(2).astype(str),
                marker=dict(size=sizes[mask], line=dict(width=0.5, color='black')),
                name=trace_name,
                legendgroup=trace_name,   # <-- this is what groups nodes+edges
                showlegend=(col == color_columns[0])
            ))

    #dropdown buttons keyed on trace.name
    buttons = []
    for col in color_columns:
        vis = []
        for tr in traces:
            # edge traces have no name → always visible
            if tr.name is None:
                vis.append(True)
            else:
                # for node traces, show only those matching this color-column
                vis.append(tr.name.startswith(f"{col} ="))
        buttons.append(dict(
            label=col,
            method="update",
            args=[{"visible": vis},
                  {"title": f"3D RMSE Network colored by {col}"}]
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.1)],
        title = {"text": f"{plot_title}: 3D RMSE Network colored by {color_columns[0]}", 
        "x": 0.5,
        "xanchor": "center"
        },
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                   aspectmode='manual', aspectratio=dict(x=6,y=6,z=1)),
        margin=dict(l=0,r=0,b=0,t=80)
    )
    fig.show()
    return fig

def plot_adj_heatmap(
  adj: np.ndarray,
  row_start: int = 0,
  row_end: int = 211,
  col_start: int = 0,
  col_end: int = 211,
  #output_html: str = 'adj_heatmap.html'
) -> go.Figure:
  """
  Create an interactive HTML heatmap of a submatrix slice of the adjacency matrix.

  Parameters:
  - adj: full adjacency NumPy array
  - row_start, row_end, col_start, col_end: slice indices
  - output_html: filename for the saved HTML

  Returns:
  - Plotly Figure (also saved to output_html)
  """
  subA = adj[row_start:row_end, col_start:col_end]

  # labels for axes
  x_labels = list(range(col_start, col_end))
  y_labels = list(range(row_end-1, row_start-1, -1))

  # create Plotly heatmap
  fig2 = go.Figure(data=go.Heatmap(
      z=subA,
      x=x_labels,
      y=y_labels,
      colorscale='Viridis',
      colorbar=dict(title='Weight')
  ))
  fig2.update_layout(
      title=f'Adjacency Submatrix {row_start}:{row_end}, {col_start}:{col_end}',
      xaxis_title='Column index',
      yaxis_title='Row index',
      width=550,
      height=400,
      margin=dict(l=50, r=50, t=50, b=50)
)


  # save to interactive HTML
  #fig2.write_html(output_html, include_plotlyjs='cdn')

  fig2.show()
  return fig2