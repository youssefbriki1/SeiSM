import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from datasets import Dataset
import torch

from tqdne.generate_waveforms import generate

def pad_with_ambient_noise(waveform, target_length=8192, noise_window=200):
    current_length = waveform.shape[1]
    pad_length = target_length - current_length
    
    if pad_length <= 0: return waveform[:, :target_length]
        
    padded_waveform = np.zeros((3, target_length))
    for c in range(3):
        channel_data = waveform[c]
        noise_mean = np.mean(channel_data[:noise_window])
        noise_std = np.std(channel_data[:noise_window])
        tail_noise = np.random.normal(loc=noise_mean, scale=noise_std + 1e-6, size=pad_length)
        padded_waveform[c] = np.concatenate([channel_data, tail_noise])
        
    return padded_waveform

def run_fast_augmentation():
    # 1. Config
    magnitudes = np.random.normal(5.2, 0.5, size=20).tolist()
    depths_km = np.random.uniform(low=5.0, high=20.0, size=5).tolist()    
    samples_per_combo = 256  
    
    original_csv_path = "/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test.csv"
    new_csv_path = "/scratch/brikiyou/ift3710/data/ceed_waveforms/events_test_augmented.csv"
    arrow_output_dir = "/scratch/brikiyou/ift3710/data/ceed_waveforms/AI4EPS___ceed/station_test/1.1.0/augmented_data"
    os.makedirs(arrow_output_dir, exist_ok=True)
    
    new_csv_rows = []
    all_generated_waveforms = []
    
    synthetic_base_time = datetime(2099, 1, 1)
    time_increment = timedelta(minutes=5)
    current_time = synthetic_base_time

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Starting H100-Optimized Generation...")

    for mag in magnitudes:
        for depth in depths_km:
            print(f"Generating {samples_per_combo} samples for Mag {mag}, Depth {depth}km...")
            temp_h5_file = f"temp_aug_mag{mag}_depth{depth}.h5"
            
            generate(
                hypocentral_distance=np.random.uniform(50, 600), 
                magnitude=mag,
                vs30=np.random.uniform(150, 400),
                hypocentre_depth=depth,
                azimuthal_gap=np.random.uniform(50, 180),
                num_samples=samples_per_combo,
                csv="", 
                outfile=temp_h5_file,
                
                batch_size=256, 
                
                edm_checkpoint="/scratch/brikiyou/ift3710/tqdne-0.2.1/tqdne/weights/edm.ckpt",
                autoencoder_checkpoint="/scratch/brikiyou/ift3710/tqdne-0.2.1/tqdne/weights/autoencoder.ckpt"
            )
            
            with h5py.File(temp_h5_file, "r") as f:
                raw_waveforms = f["waveforms"][:] 
                for i in range(samples_per_combo):
                    padded_wave = pad_with_ambient_noise(raw_waveforms[i], target_length=8192)
                    event_time_str = current_time.isoformat()
                    
                    begin_time_dt = current_time - timedelta(seconds=28)
                    end_time_dt = begin_time_dt + timedelta(seconds=120)

                    new_csv_rows.append({
                        "begin_time": begin_time_dt.isoformat(),
                        "depth_km": depth,
                        "end_time": end_time_dt.isoformat(),
                        "event_id": f"aug_m{mag:.2f}_d{depth:.2f}_{i}",
                        "event_time": current_time.isoformat(),
                        "event_time_index": 2800.0,    # Average pick index from your original data
                        "latitude": 37.0,              # Dummy central California coord
                        "longitude": -120.0,           # Dummy central California coord
                        "magnitude": mag,
                        "magnitude_type": "SYN",       # Tagging type as Synthetic
                        "nt": 8192.0,                  # Your padded waveform length
                        "nx": 1.0,                     # Single generated station
                        "sampling_rate": 100, 
                        "source": "TQDNE_AUG" 
                    })           
                             
                    all_generated_waveforms.append(padded_wave)
                    current_time += time_increment
            
            os.remove(temp_h5_file)

    print("\nMerging metadata into new CSV...")
    df_original = pd.read_csv(original_csv_path)
    df_augmented = pd.DataFrame(new_csv_rows)
    df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
    df_combined.to_csv(new_csv_path, index=False)
    
    print("Converting to Arrow format...")
    hf_dataset = Dataset.from_dict({
        "data": all_generated_waveforms,
        "event_time": [row["event_time"] for row in new_csv_rows]
    })
    
    hf_dataset.save_to_disk(arrow_output_dir)
    print(f"Augmentation complete! Added {len(new_csv_rows)} samples.")

if __name__ == "__main__":
    run_fast_augmentation()