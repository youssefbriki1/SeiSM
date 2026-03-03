import pandas as pd

def main():
    input_file = "data/1970-2021_11_EARTH_final_with_patchnum.csv"
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    years = df['onlydate'].str[:4].astype(int)
    
    #Training dataset
    filtered_df = df[(years >= 1970) & (years <= 2010)]
    print(f"Saving training dataset ({len(filtered_df)} rows)...")
    filtered_df.to_csv("data/training_data.csv", index=False)

    #Validation dataset
    filtered_df = df[(years >= 2002) & (years <= 2021)]
    print(f"Saving testing dataset ({len(filtered_df)} rows)...")
    filtered_df.to_csv("data/testing_data.csv", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
