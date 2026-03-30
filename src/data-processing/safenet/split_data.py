import os
from pathlib import Path
import pandas as pd

def main():
    DATA_DIR = Path(__file__).resolve().parent / 'dataset'
    OUTPUT_DIR = Path(__file__).resolve().parent / 'data'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_file = os.path.join(DATA_DIR, "1970-2021_11_EARTH_final_with_patchnum.csv")
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    years = df['onlydate'].str[:4].astype(int)
    
    #Training dataset, will generate feature-engineered data for 1979 - 2010
    filtered_df = df[(years >= 1970) & (years <= 2010)]
    print(f"Saving training dataset ({len(filtered_df)} rows)...")
    filtered_df.to_csv(os.path.join(OUTPUT_DIR, "training_data.csv"), index=False)

    #Validation dataset, will generate feature-engineered data for 2011 - 2021
    filtered_df = df[(years >= 2002) & (years <= 2021)]
    print(f"Saving testing dataset ({len(filtered_df)} rows)...")
    filtered_df.to_csv(os.path.join(OUTPUT_DIR, "testing_data.csv"), index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
