"""
One-time script to backfill existing training runs into W&B.

Walks all output directories under OUTPUTS_DIR/seed_experiment/5_rivers,
reads losses.json and eval_fw*/rmse_test.json, reconstructs config from
directory names, and logs each run to wandb.

Usage (from Colab or local):
    python wandb_backfill.py
"""

import json
import re
import os
import sys
import numpy as np
from pathlib import Path

import wandb

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR


def parse_model_dir_name(name):
    """Extract seed from model directory name like '...._s42' or '..._s3141'."""
    m = re.search(r'_s(\d+)$', name)
    return int(m.group(1)) if m else None


def parse_graph_label(graph_label):
    """Parse the graph_type subdirectory name into config-like tags."""
    tags = [graph_label]
    config = {"graph_label": graph_label}

    # Graph type
    if graph_label.startswith("feature_distance"):
        config["graph_type"] = "feature_distance"
        tags.append("feature_distance")
        m = re.search(r'_r(\d+)', graph_label)
        if m:
            config["fd_radius"] = int(m.group(1)) / 100.0
        if "_wm" in graph_label:
            m = re.search(r'_wm(\d+)', graph_label)
            if m:
                config["fd_weight_max"] = int(m.group(1)) / 10.0
        if "_taccari" in graph_label:
            config["fd_taccari"] = True
    elif graph_label.startswith("shortest_path"):
        config["graph_type"] = "shortest_path"
        tags.append("shortest_path")
    elif graph_label.startswith("rf_cutoff"):
        config["graph_type"] = "rf"
        config["weight_mode"] = "cutoff"
        tags.append("rf")
    elif graph_label.startswith("rf"):
        config["graph_type"] = "rf"
        tags.append("rf")
    elif graph_label.startswith("geolayer"):
        config["graph_type"] = "geolayer"
        tags.append("geolayer")
    elif graph_label.startswith("default"):
        config["graph_type"] = "default"
        tags.append("default")
    elif graph_label.startswith("adaptive"):
        config["graph_type"] = "adaptive"
        config["build_adj"] = True
        tags.append("adaptive")
    elif graph_label.startswith("mixed"):
        config["graph_type"] = "mixed"
        tags.append("mixed")
    else:
        config["graph_type"] = graph_label

    # Ablation flags
    if "_no_pumps" in graph_label:
        config["remove_pumps"] = True
        tags.append("no_pumps")
    if "_no_rivers" in graph_label:
        config["remove_rivers"] = True
        tags.append("no_rivers")
    if "_no_evap" in graph_label:
        config["remove_evap"] = True
        tags.append("no_evap")
    if "_no_precip" in graph_label:
        config["remove_precip"] = True
        tags.append("no_precip")

    # Pump weight config
    if "_pump_thiem" in graph_label:
        config["pump_weight_source"] = "thiem"
        m = re.search(r'_pump_thiem_m([\d.]+)', graph_label)
        if m:
            config["thiem_multiplier"] = float(m.group(1))
        tags.append("pump_thiem")
    elif "_pump_coherence" in graph_label:
        config["pump_weight_source"] = "coherence"
        if "_90_365d" in graph_label:
            config["coherence_band"] = "90_365d"
        elif "_gt365d" in graph_label:
            config["coherence_band"] = "gt365d"
        tags.append("pump_coherence")

    # Split
    if "_split70" in graph_label:
        config["val_split"] = True
        config["train_pct"] = 70
        tags.append("split70")

    # Multi-support
    if "_multi_support" in graph_label:
        config["multi_support"] = True
        tags.append("multi_support")

    # Node dropout
    if "_node_dropout" in graph_label:
        config["node_dropout"] = True
        tags.append("node_dropout")

    return config, tags


def load_run_data(run_dir):
    """Load losses and per-fw test RMSE from a run directory."""
    data = {}

    # Losses
    losses_path = run_dir / "losses.json"
    if losses_path.exists():
        with open(losses_path) as f:
            data["losses"] = json.load(f)

    # Per-fw test RMSE
    data["test_rmse"] = {}
    for fw in (1, 2, 3):
        rmse_path = run_dir / f"eval_fw{fw}" / "rmse_test.json"
        if rmse_path.exists():
            with open(rmse_path) as f:
                node_rmses = np.array(json.load(f))
            data["test_rmse"][fw] = {
                "mean": float(np.mean(node_rmses)),
                "std": float(np.std(node_rmses)),
                "per_node": node_rmses,
            }

    return data


def backfill_run(run_dir, experiment_name, graph_label, is_ablation=False):
    """Log a single existing run to wandb."""
    model_dir_name = run_dir.name
    seed = parse_model_dir_name(model_dir_name)
    if seed is None:
        print(f"  SKIP (no seed): {run_dir}")
        return False

    run_data = load_run_data(run_dir)
    if not run_data.get("test_rmse"):
        print(f"  SKIP (no test RMSE): {run_dir}")
        return False

    config, tags = parse_graph_label(graph_label)
    config["seed"] = seed
    config["experiment_name"] = experiment_name
    config["run_dir"] = str(run_dir)
    config["model_dir"] = model_dir_name
    config["is_ablation"] = is_ablation
    tags.append(f"seed_{seed}")
    if is_ablation:
        tags.append("ablation")

    # Parse N_PUMPS from dir name
    m = re.search(r'N_PUMPS\s*_(\d+)', model_dir_name)
    if m:
        config["n_pumps_connected"] = int(m.group(1))

    # Parse node_dropout from dir name
    if "_node_dropout_w" in model_dir_name:
        config["node_dropout"] = True
        m = re.search(r'_node_dropout_w(\d+)', model_dir_name)
        if m:
            config["node_dropout_warmup"] = int(m.group(1))

    run = wandb.init(
        project="groundwater-flow-gnn",
        name=f"{graph_label}_s{seed}",
        config=config,
        tags=tags,
        reinit=True,
    )

    # Log per-epoch losses
    losses = run_data.get("losses", {})
    for window_key in sorted(losses.get("train_losses", {}).keys()):
        fw_num = window_key.replace("window_", "")
        train_entries = losses["train_losses"].get(window_key, [])
        eval_entries = losses["eval_losses"].get(window_key, [])

        for i, (tr, ev) in enumerate(zip(train_entries, eval_entries)):
            wandb.log({
                f"fw{fw_num}/train_loss": tr["loss"],
                f"fw{fw_num}/eval_loss": ev["loss"],
                f"fw{fw_num}/epoch": tr["epoch"],
            })

    # Log test RMSE per fw
    for fw, rmse_data in run_data["test_rmse"].items():
        wandb.log({
            f"test/fw{fw}_rmse_mean": rmse_data["mean"],
            f"test/fw{fw}_rmse_std": rmse_data["std"],
        })

    # Summary: best fw
    best_fw = min(run_data["test_rmse"].keys(),
                  key=lambda k: run_data["test_rmse"][k]["mean"])
    best = run_data["test_rmse"][best_fw]
    run.summary["best_fw"] = best_fw
    run.summary["best_rmse_mean"] = best["mean"]
    run.summary["best_rmse_std"] = best["std"]

    wandb.finish()
    return True


def main():
    base = OUTPUTS_DIR / "seed_experiment"
    experiment_dirs = [d for d in base.iterdir() if d.is_dir()]

    total_logged = 0
    total_skipped = 0

    for exp_dir in sorted(experiment_dirs):
        experiment_name = exp_dir.name
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}")

        # Non-ablation graph-type dirs
        for graph_dir in sorted(exp_dir.iterdir()):
            if not graph_dir.is_dir():
                continue
            graph_label = graph_dir.name

            if graph_label == "ablation":
                # Process ablation subdirectories
                for abl_dir in sorted(graph_dir.iterdir()):
                    if not abl_dir.is_dir():
                        continue
                    abl_label = abl_dir.name
                    print(f"\n  [ablation] {abl_label}")
                    for run_dir in sorted(abl_dir.iterdir()):
                        if not run_dir.is_dir():
                            continue
                        if backfill_run(run_dir, experiment_name, abl_label, is_ablation=True):
                            total_logged += 1
                        else:
                            total_skipped += 1
            else:
                print(f"\n  {graph_label}")
                for run_dir in sorted(graph_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    if backfill_run(run_dir, experiment_name, graph_label):
                        total_logged += 1
                    else:
                        total_skipped += 1

    print(f"\n{'='*60}")
    print(f"Backfill complete: {total_logged} runs logged, {total_skipped} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
