from datasets import load_dataset
import pandas as pd
import os

save_dir = "/scratch/brikiyou/ift3710/data/ceed_waveforms"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "events_train.csv")

print("Downloading from Hugging Face...")
df = pd.read_csv("hf://datasets/AI4EPS/CEED/events_train.csv")
df.to_csv(save_path, index=False)

print(f"Success! Saved permanently to: {save_path}")


ceed_test = load_dataset(
    "AI4EPS/CEED", 
    name="station_test", 
    split="test", 
    trust_remote_code=True,
    cache_dir="/scratch/brikiyou/ift3710/data/ceed_waveforms"
)

print("\nSuccess! Here is the dataset structure:")
print(ceed_test)

first_sample = ceed_test[0]
print(f"Waveform shape: {len(first_sample['data'])} channels, {len(first_sample['data'][0])} timesteps")

