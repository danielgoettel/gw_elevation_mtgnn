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

    # Weight mode — match only the mode word, not trailing _SAMELAYER etc.
    wm_match = re.search(r'_WM[: _]+(fixed|variable)', name)
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


def select_best_fw(df, piezo_names):
    """For each (Variant, Seed), keep only the F_w with the lowest Overall RMSE."""
    best_rows = []
    for (variant, seed), grp in df.groupby(['Variant', 'Seed']):
        best_idx = grp['Overall RMSE'].idxmin()
        best_rows.append(grp.loc[best_idx])
    return pd.DataFrame(best_rows).reset_index(drop=True)


def classify_nodes(df_best, piezo_names):
    """
    Classify each piezometer as Consistent Low / Moderate / High / Model-Dependent.

    Returns
    -------
    node_stats : DataFrame with columns [name, mean_rmse, cv_variant, category]
        Sorted by mean_rmse descending (highest RMSE first).
    """
    variants = sorted(df_best['Variant'].unique())

    # Per-variant mean RMSE for each node
    variant_means = {}
    for v in variants:
        sub = df_best[df_best['Variant'] == v]
        variant_means[v] = sub[piezo_names].mean(axis=0).values

    variant_stack = np.array(list(variant_means.values()))  # (n_variants, n_nodes)
    node_means = variant_stack.mean(axis=0)
    node_cv = variant_stack.std(axis=0) / (variant_stack.mean(axis=0) + 1e-12)

    categories = []
    for i in range(len(piezo_names)):
        if node_cv[i] > 0.15:
            categories.append('Model-Dependent')
        elif node_means[i] < 10:
            categories.append('Consistent Low')
        elif node_means[i] > 30:
            categories.append('Consistent High')
        else:
            categories.append('Consistent Moderate')

    node_stats = pd.DataFrame({
        'name': piezo_names,
        'mean_rmse': node_means,
        'cv_variant': node_cv * 100,  # as percentage
        'category': categories,
    })
    # Per-variant columns
    for v in variants:
        node_stats[f'rmse_{v}'] = variant_means[v]

    # Sort by mean RMSE descending (highest first)
    node_stats = node_stats.sort_values('mean_rmse', ascending=False).reset_index(drop=True)
    return node_stats


def write_comparison_xlsx(output_path, df_best, node_stats, piezo_names):
    """
    Write a formatted Excel workbook with:
      Sheet 1 — Per-Node RMSE Comparison (columns sorted by mean RMSE descending)
      Sheet 2 — Summary (category counts, variant stats, classification criteria)
    """
    # Sort piezo columns by mean RMSE descending
    sorted_names = node_stats['name'].tolist()  # already sorted desc

    # Build the Per-Node sheet: header rows + data rows
    meta_cols = ['Variant', 'Seed', 'F_w', 'Overall RMSE']
    col_order = meta_cols + sorted_names

    # Row 1: Category labels
    # Row 2: Mean RMSE per node
    # Row 3: Column headers
    # Rows 4+: Data

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = 'Per-Node RMSE Comparison'

    cat_colors = {
        'Consistent Low': '92D050',
        'Consistent Moderate': 'FFC000',
        'Consistent High': 'FF6666',
        'Model-Dependent': '9BC2E6',
    }

    # --- Row 1: Category ---
    ws.cell(1, 1, 'Category')
    ws.cell(1, 1).font = Font(bold=True)
    for ci, pname in enumerate(sorted_names):
        cat = node_stats.loc[node_stats['name'] == pname, 'category'].iloc[0]
        cell = ws.cell(1, len(meta_cols) + 1 + ci, cat)
        cell.fill = PatternFill('solid', fgColor=cat_colors.get(cat, 'FFFFFF'))
        cell.font = Font(size=8)

    # --- Row 2: Mean RMSE ---
    ws.cell(2, 1, 'Mean RMSE')
    ws.cell(2, 1).font = Font(bold=True)
    for ci, pname in enumerate(sorted_names):
        mean_val = node_stats.loc[node_stats['name'] == pname, 'mean_rmse'].iloc[0]
        ws.cell(2, len(meta_cols) + 1 + ci, round(mean_val, 2))

    # --- Row 3: Headers ---
    for ci, col in enumerate(col_order):
        cell = ws.cell(3, ci + 1, col)
        cell.font = Font(bold=True)

    # --- Row 4+: Data rows (sorted by Variant, Seed) ---
    df_sorted = df_best.sort_values(['Variant', 'Seed']).reset_index(drop=True)
    for ri, (_, row) in enumerate(df_sorted.iterrows()):
        for ci, col in enumerate(col_order):
            val = row.get(col)
            if isinstance(val, float):
                val = round(val, 2)
            ws.cell(4 + ri, ci + 1, val)

    # Auto-fit column widths for meta columns
    for ci in range(1, len(meta_cols) + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 14

    # --- Summary sheet ---
    ws2 = wb.create_sheet('Summary')

    # Category summary
    ws2.cell(1, 1, 'Category').font = Font(bold=True)
    ws2.cell(1, 2, 'Count').font = Font(bold=True)
    ws2.cell(1, 3, 'Mean RMSE').font = Font(bold=True)
    ws2.cell(1, 4, 'Min RMSE').font = Font(bold=True)
    ws2.cell(1, 5, 'Max RMSE').font = Font(bold=True)

    for ri, cat in enumerate(['Consistent Low', 'Consistent Moderate', 'Consistent High', 'Model-Dependent']):
        sub = node_stats[node_stats['category'] == cat]
        ws2.cell(2 + ri, 1, cat)
        ws2.cell(2 + ri, 1).fill = PatternFill('solid', fgColor=cat_colors[cat])
        ws2.cell(2 + ri, 2, len(sub))
        if len(sub) > 0:
            ws2.cell(2 + ri, 3, round(sub['mean_rmse'].mean(), 2))
            ws2.cell(2 + ri, 4, round(sub['mean_rmse'].min(), 2))
            ws2.cell(2 + ri, 5, round(sub['mean_rmse'].max(), 2))

    # Variant summary
    row_offset = 7
    ws2.cell(row_offset, 1, 'Variant').font = Font(bold=True)
    ws2.cell(row_offset, 2, 'Mean RMSE').font = Font(bold=True)
    ws2.cell(row_offset, 3, 'StdDev').font = Font(bold=True)

    variants = sorted(df_best['Variant'].unique())
    for vi, v in enumerate(variants):
        sub = df_best[df_best['Variant'] == v]
        ws2.cell(row_offset + 1 + vi, 1, v)
        ws2.cell(row_offset + 1 + vi, 2, round(sub['Overall RMSE'].mean(), 2))
        ws2.cell(row_offset + 1 + vi, 3, round(sub['Overall RMSE'].std(), 2))

    # Classification criteria
    row_offset2 = row_offset + 2 + len(variants)
    criteria = [
        'Classification Criteria:',
        'Model-Dependent: CV across graph-type means > 15%',
        'Consistent Low: mean RMSE < 10 cm',
        'Consistent High: mean RMSE > 30 cm',
        'Consistent Moderate: everything else',
        f'Based on: {len(variants)} variants x {len(df_best["Seed"].unique())} seeds = {len(df_best)} runs, best F_w per run',
    ]
    for ci, text in enumerate(criteria):
        ws2.cell(row_offset2 + ci, 1, text)

    # Column widths
    ws2.column_dimensions['A'].width = 60
    for c in ['B', 'C', 'D', 'E']:
        ws2.column_dimensions[c].width = 12

    wb.save(str(output_path))
    return output_path


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
        existing = pd.read_excel(str(output_path), sheet_name='Per-Node RMSE Comparison',
                                 header=2)  # header on row 3
        df = pd.concat([existing, df], ignore_index=True)
        # Drop exact duplicates (same Variant + Seed + F_w)
        df = df.drop_duplicates(subset=['Variant', 'Seed', 'F_w'], keep='last')
        df = df.sort_values(['Variant', 'Seed', 'F_w']).reset_index(drop=True)
        print(f"Appending to existing file ({len(existing)} existing + {len(rows)} new rows)")

    # Select best F_w per (Variant, Seed), classify nodes, write formatted output
    df_best = select_best_fw(df, piezo_names)
    node_stats = classify_nodes(df_best, piezo_names)
    write_comparison_xlsx(output_path, df_best, node_stats, piezo_names)

    # Print summary
    print(f"\nWrote {len(df_best)} rows (best F_w per variant×seed) to {output_path}")
    print(f"Variants: {sorted(df_best['Variant'].unique())}")
    print(f"Seeds: {sorted(df_best['Seed'].unique())}")
    print(f"Overall RMSE: mean={df_best['Overall RMSE'].mean():.2f}, "
          f"std={df_best['Overall RMSE'].std():.2f}")
    print(f"\nNode classification:")
    for cat in ['Consistent Low', 'Consistent Moderate', 'Consistent High', 'Model-Dependent']:
        n = (node_stats['category'] == cat).sum()
        if n > 0:
            sub = node_stats[node_stats['category'] == cat]
            print(f"  {cat}: {n} nodes (RMSE {sub['mean_rmse'].min():.1f} – {sub['mean_rmse'].max():.1f})")
        else:
            print(f"  {cat}: 0 nodes")


if __name__ == '__main__':
    main()
