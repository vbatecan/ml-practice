import pandas as pd
import glob
import os

def combine_csvs(output_file="combined_data.csv"):
    # Find all CSV files matching the pattern
    csv_files = glob.glob("racing_data_*.csv")
    
    if not csv_files:
        print("No racing data CSV files found.")
        return

    print(f"Found {len(csv_files)} files.")

    # Read the first file to get the reference header
    try:
        reference_df = pd.read_csv(csv_files[0])
        reference_header = list(reference_df.columns)
        print(f"Reference header established from {csv_files[0]}")
    except Exception as e:
        print(f"Error reading reference file {csv_files[0]}: {e}")
        return

    valid_dfs = []
    
    # Iterate through all files
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            header = list(df.columns)
            
            if header == reference_header:
                valid_dfs.append(df)
                print(f"Checked {file}: INTEGRITY CONFIRMED")
            else:
                print(f"Checked {file}: INTEGRITY FAILED (Header mismatch)")
                print(f"Expected: {reference_header}")
                print(f"Found:    {header}")
        except Exception as e:
            print(f"Checked {file}: ERROR reading file ({e})")

    if valid_dfs:
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccess! {len(valid_dfs)} files combined into '{output_file}'.")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("\nNo valid files to combine.")

if __name__ == "__main__":
    combine_csvs()
