def export_combined_stresses_by_type(pstore, stress_keywords, output_base="exported_stresses"):
    """
    Export combined stress time series for each keyword to a single CSV file.

    Parameters
    ----------
    pstore : pastastore.PastaStore
        Your open PastaStore connection.
    stress_keywords : list of str
        Substrings to match in stress names (e.g., ['riv', 'precip', 'well']).
    output_base : str
        Output directory where combined CSVs will be saved.
    """
    import os
    import pandas as pd

    os.makedirs(output_base, exist_ok=True)
    all_stress_names = pstore.stresses.index

    for keyword in stress_keywords:
        matched = [name for name in all_stress_names if keyword in name.lower()]
        if not matched:
            print(f"⚠️ No stresses matched keyword '{keyword}'")
            continue

        combined_df = pd.DataFrame()

        for name in matched:
            try:
                series = pstore.get_stresses(name)
                df = pd.DataFrame(series)
                df.columns = [name]  # Ensure unique column per stress series
                combined_df = combined_df.join(df, how="outer")
                print(f"✅ Added: {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")

        # Save one combined CSV per keyword
        out_path = os.path.join(output_base, f"{keyword}_combined.csv")
        combined_df.to_csv(out_path)
        print(f"📄 Saved combined CSV: {out_path}")


from util import get_pastastore

str = get_pastastore('stresses', 'pas')
export_combined_stresses_by_type(str, stress_keywords=["rh"])
