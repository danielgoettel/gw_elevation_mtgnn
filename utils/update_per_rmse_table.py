"""
Build a per-node RMSE comparison table from seed experiment results.

Usage:
    python -m utils.update_per_rmse_table <input_folder> <output_folder>

Example:
    python -m utils.update_per_rmse_table outputs_Independent_Study/seed_experiment outputs_Independent_Study/seed_experiment

The script:
  1. Walks <input_folder>/{graph_type}/{run_folder}/eval_fw{N}/rmse_test.json
  2. Parses variant, seed, F_w from the folder structure / name
  3. Loads piezometer names from the preprocessed column_names_real.txt
  4. Writes per_node_rmse_comparison.xlsx to <output_folder>
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run_folder(run_path, graph_type):
    """Extract seed from run folder name.

    Folder names look like:
        MTGNN_fw3_default_W5_N_Piezo _3_FIM _1_WM _fixed_N_PUMPS _4_s42
    """
    name = run_path.name
    seed_match = re.search(r'_s(\d+)$', name)
    seed = int(seed_match.group(1)) if seed_match else None

    # Weight mode
    wm_match = re.search(r'_WM[: _]+(\w+)', name)
    weight_mode = wm_match.group(1) if wm_match else 'fixed'

    # Layer constrain
    layer_constrain = 'SAMELAYER' in name

    # Build variant label matching the convention in rmse_geology_analysis
    if graph_type == 'rf':
        parts = ['rf']
        if weight_mode == 'variable':
            parts.append('variable')
        if layer_constrain:
            parts.append('geolayer-sep')
        variant = ' + '.join(parts) if len(parts) > 1 else 'rf'
    else:
        variant = graph_type

    return variant, seed


def collect_rmse_rows(input_folder, piezo_names):
    """Walk the input folder and collect one row per (variant, seed, fw)."""
    input_folder = Path(input_folder)
    rows = []

    # Detect structure: does input_folder contain graph_type subdirs, or run folders directly?
    # If any immediate child has an eval_fw* subfolder, treat input_folder as a single graph_type.
    has_run_folders = any(
        (child / 'eval_fw1').exists() or (child / 'eval_fw3').exists()
        for child in input_folder.iterdir() if child.is_dir()
    )

    if has_run_folders:
        # Flat mode: input_folder IS the graph_type folder
        gt_dirs = [(input_folder, input_folder.name)]
    else:
        # Nested mode: input_folder contains graph_type subdirectories
        gt_dirs = [(d, d.name) for d in sorted(input_folder.iterdir()) if d.is_dir()]

    for gt_dir, graph_type in gt_dirs:
        # Iterate over run folders
        for run_dir in sorted(gt_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            variant, seed = parse_run_folder(run_dir, graph_type)

            # Find eval_fw* subfolders
            for eval_dir in sorted(run_dir.glob('eval_fw*')):
                rmse_file = eval_dir / 'rmse_test.json'
                if not rmse_file.exists():
                    continue

                fw_match = re.search(r'eval_fw(\d+)', eval_dir.name)
                fw = int(fw_match.group(1)) if fw_match else None

                with open(rmse_file) as f:
                    rmse_values = json.load(f)

                # Truncate or pad to match piezo_names length
                n = len(piezo_names)
                rmse_values = rmse_values[:n]

                overall = float(np.sqrt(np.mean(np.array(rmse_values) ** 2)))

                row = {
                    'Variant': variant,
                    'Seed': seed,
                    'F_w': fw,
                    'Overall RMSE': round(overall, 4),
                }
                for name, val in zip(piezo_names, rmse_values):
                    row[name] = round(val, 4)

                rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Build per-node RMSE table from seed experiment results.')
    parser.add_argument('input_folder', help='Root folder with graph_type subdirectories (e.g., seed_experiment/)')
    parser.add_argument('output_folder', help='Folder to write the output Excel file')
    parser.add_argument('--piezo-names', default=None,
                        help='Path to column_names_real.txt (auto-detected if not given)')
    parser.add_argument('--output-name', default='per_node_rmse_comparison.xlsx',
                        help='Output filename (default: per_node_rmse_comparison.xlsx)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing file instead of overwriting')
    args = parser.parse_args()

    # Auto-detect piezo names
    if args.piezo_names:
        names_path = Path(args.piezo_names)
    else:
        # Try to find relative to script location
        script_dir = Path(__file__).resolve().parent.parent
        names_path = script_dir / 'data' / 'preprocessed_Independent_Study' / 'column_names_real.txt'
        if not names_path.exists():
            # Try common Colab/Drive path
            names_path = Path('data/preprocessed_Independent_Study/column_names_real.txt')

    if not names_path.exists():
        print(f"ERROR: Could not find column_names_real.txt at {names_path}")
        print("Use --piezo-names to specify the path explicitly.")
        sys.exit(1)

    with open(names_path) as f:
        piezo_names = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(piezo_names)} piezometer names from {names_path}")

    # Collect RMSE rows
    rows = collect_rmse_rows(args.input_folder, piezo_names)
    if not rows:
        print(f"WARNING: No rmse_test.json files found under {args.input_folder}")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values(['Variant', 'Seed', 'F_w']).reset_index(drop=True)

    # Write output
    output_path = Path(args.output_folder) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.append and output_path.exists():
        existing = pd.read_excel(str(output_path))
        df = pd.concat([existing, df], ignore_index=True)
        # Drop exact duplicates (same Variant + Seed + F_w)
        df = df.drop_duplicates(subset=['Variant', 'Seed', 'F_w'], keep='last')
        df = df.sort_values(['Variant', 'Seed', 'F_w']).reset_index(drop=True)
        print(f"Appending to existing file ({len(existing)} existing + {len(rows)} new rows)")

    df.to_excel(str(output_path), index=False, sheet_name='Per-Node RMSE Comparison')

    print(f"\nWrote {len(df)} rows to {output_path}")
    print(f"Variants: {sorted(df['Variant'].unique())}")
    print(f"Seeds: {sorted(df['Seed'].unique())}")
    print(f"F_w values: {sorted(df['F_w'].unique())}")
    print(f"Overall RMSE: mean={df['Overall RMSE'].mean():.2f}, "
          f"std={df['Overall RMSE'].std():.2f}")


if __name__ == '__main__':
    main()
