import wandb
import os

def download_wandb_checkpoints():
    api = wandb.Api()
    entity = "ift3710-ai-slop"
    project = "safenet-full"
    run_id = "f1qhjpik"
    run_path = f"{entity}/{project}/{run_id}"
    
    print(f"Connecting to W&B run: {run_path}...")
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Error accessing the run. Make sure it exists and you have access permissions.\nDetails: {e}")
        return

    download_dir = "/scratch/brikiyou/ift3710/checkpoints/images"
    os.makedirs(download_dir, exist_ok=True)

    print("Scanning files in the run...")
    files_downloaded = 0
    
    for file in run.files():
        if file.name.startswith("checkpoints/"):
            print(f"Downloading {file.name}...")
            file.download(root=download_dir, replace=True)
            files_downloaded += 1

    if files_downloaded > 0:
        print(f"\nSuccessfully downloaded {files_downloaded} file(s) to '{download_dir}/checkpoints/'.")
    else:
        print("\nNo files found in the 'checkpoints' directory for this run.")

if __name__ == "__main__":
    download_wandb_checkpoints()