import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Import models
from models import QuakeWaveMamba2, SafeNetSSM, SeiSM

def evaluate_fusion(model, dataloader, device, num_classes=4, desc="Evaluating"):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for waves, safe_in, targets in tqdm(dataloader, desc=desc):
            waves = waves.to(device)
            safe_in = {k: v.to(device) for k, v in safe_in.items()}
            targets = targets.to(device) # (B, Patches)
            
            # Forward pass: returns (B, Patches, 4)
            outputs = model(waves, safe_in)
            
            # Flatten to (B*Patches, 4) and (B*Patches)
            outputs_flat = outputs.view(-1, num_classes)
            targets_flat = targets.view(-1)
            
            preds = torch.argmax(outputs_flat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets_flat.cpu().numpy())
            
    # Calculate Metrics
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        "accuracy": accuracy,
        "f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate the MLP Fusion (SeiSM) model")
    parser.add_argument("--mamba_weights", type=str, default="checkpoints/best_mamba2_adamw_waveform.pth", help="Path to pre-trained Mamba2 weights")
    parser.add_argument("--safenet_weights", type=str, default="checkpoints/safenet_ssm.pth", help="Path to pre-trained SafeNetSSM weights")
    parser.add_argument("--seism_mlp_weights", type=str, default="checkpoints/best_seism_mlp.pth", help="Path to the trained fused model MLP weights")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_patches", type=int, default=85)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. Initialize and Load Weights
    # ==========================================
    print("Initializing models...")
    mamba = QuakeWaveMamba2(in_channels=3, d_model=128, d_state=64, n_layers=4, headdim=32)
    safenet = SafeNetSSM(num_classes=4, num_patches=args.num_patches, d_model=128)
    fused_model = SeiSM(mamba_model=mamba, safenet_model=safenet, mlp_hidden_dim=32, final_classes=4).to(device)

    # Load Base Models
    if os.path.exists(args.mamba_weights):
        mamba.load_state_dict(torch.load(args.mamba_weights, map_location="cpu"))
        print(f"Loaded Mamba weights from {args.mamba_weights}")
    else:
        print(f"WARNING: Mamba weights not found at {args.mamba_weights}.")
        
    if os.path.exists(args.safenet_weights):
        safenet.load_state_dict(torch.load(args.safenet_weights, map_location="cpu"))
        print(f"Loaded SafeNet weights from {args.safenet_weights}")
    else:
        print(f"WARNING: SafeNet weights not found at {args.safenet_weights}.")

    # Load the Trained MLP
    if os.path.exists(args.seism_mlp_weights):
        fused_model.mlp.load_state_dict(torch.load(args.seism_mlp_weights, map_location=device))
        print(f"Loaded SeiSM MLP weights from {args.seism_mlp_weights}")
    else:
        print(f"ERROR: Trained MLP weights not found at {args.seism_mlp_weights}. Cannot evaluate.")
        return

    # Set to eval mode
    fused_model.eval()

    # ==========================================
    # 2. Data Setup (TODO: Replace with actual dataloader)
    # ==========================================
    # Replace Dummy Dataset with your actual Test Dataset
    num_samples = 64
    
    class DummyMultimodalDataset(torch.utils.data.Dataset):
        def __len__(self): return num_samples
        def __getitem__(self, idx):
            waveforms = torch.randn(3, 8192)
            safe_inputs = {
                "catalog": torch.randn(10, 86, 282),
                "maps": torch.randn(10, args.num_patches, 50, 50, 5)
            }
            targets = torch.randint(0, 4, (args.num_patches,))
            return waveforms, safe_inputs, targets

    test_dataset = DummyMultimodalDataset()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 3. Evaluate
    # ==========================================
    print("\nStarting evaluation on Test Set...")
    metrics = evaluate_fusion(fused_model, test_loader, device, desc="Testing")
    
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1 Score:   {metrics['f1']:.4f}")
    print(f"Macro Precision:  {metrics['precision']:.4f}")
    print(f"Macro Recall:     {metrics['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*40)

if __name__ == "__main__":
    main()
