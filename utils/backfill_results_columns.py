"""
backfill_results_columns.py
───────────────────────────
Adds new columns (Variant, FD Radius, FD Weight Max, FD Pump Weight,
FD River Weight, RF Cutoff, RF Min Conn) to existing seed_experiment_results.xlsx
and per_node_results.xlsx files.

Old rows that predate these columns get empty strings.
New rows written by train_model.py will already have them populated.

Usage (run once, with the Excel files closed):
    python utils/backfill_results_columns.py
"""
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs_Independent_Study"

NEW_SEED_COLS = ["Variant", "FD Radius", "FD Weight Max", "FD Pump Weight",
                 "FD River Weight", "RF Cutoff", "RF Min Conn"]

# In seed_experiment_results the new cols sit after "VIM Min"
SEED_INSERT_AFTER = "VIM Min"

# In per_node_results the new cols sit after "RF Weight Range"
NODE_INSERT_AFTER = "RF Weight Range"


def _insert_cols_after(df: pd.DataFrame, after: str, new_cols: list) -> pd.DataFrame:
    """Insert new_cols (filled with '') immediately after column `after`."""
    existing_new = [c for c in new_cols if c in df.columns]
    missing_new = [c for c in new_cols if c not in df.columns]
    if not missing_new:
        print("  All columns already present, skipping.")
        return df

    if after in df.columns:
        pos = df.columns.get_loc(after) + 1
    else:
        pos = len(df.columns)  # append at end if anchor not found

    for i, col in enumerate(missing_new):
        df.insert(pos + i, col, "")

    print(f"  Inserted {missing_new} after '{after}'.")
    return df


def backfill(xlsx_path: Path, insert_after: str):
    if not xlsx_path.exists():
        print(f"  Not found: {xlsx_path}")
        return

    print(f"\nProcessing: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    df = _insert_cols_after(df, insert_after, NEW_SEED_COLS)

    # Overwrite with updated DataFrame
    backup = xlsx_path.with_suffix(".xlsx.bak")
    if backup.exists():
        backup.unlink()
    xlsx_path.rename(backup)
    df.to_excel(xlsx_path, index=False)
    print(f"  Saved. Backup at {backup.name}")


def main():
    # Find all seed_experiment_results and per_node_results files under OUTPUTS_DIR
    for seed_file in OUTPUTS_DIR.rglob("seed_experiment_results.xlsx"):
        backfill(seed_file, SEED_INSERT_AFTER)

    for node_file in OUTPUTS_DIR.rglob("per_node_results.xlsx"):
        backfill(node_file, NODE_INSERT_AFTER)

    print("\nDone.")


if __name__ == "__main__":
    main()
