import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# Import models
from models import QuakeWaveMamba2, SafeNetSSM, SeiSM

def evaluate_fusion(model, dataloader, criterion, device, num_classes=4, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
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
            
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs_flat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets_flat.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    # Calculate Metrics
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": macro_f1
    }


def main():
    parser = argparse.ArgumentParser(description="Train MLP Fusion (SeiSM) over frozen Mamba and SafeNetSSM")
    parser.add_argument("--mamba_weights", type=str, default="checkpoints/best_mamba2_adamw_waveform.pth", help="Path to pre-trained Mamba2 weights")
    parser.add_argument("--safenet_weights", type=str, default="checkpoints/safenet_ssm.pth", help="Path to pre-trained SafeNetSSM weights")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_seism_mlp.pth", help="Path to save the fused model MLP weights")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_patches", type=int, default=85)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. Initialize and Load Pre-trained Weights
    # ==========================================
    print("Initializing base models...")
    mamba = QuakeWaveMamba2(in_channels=3, d_model=128, d_state=64, n_layers=4, headdim=32)
    safenet = SafeNetSSM(num_classes=4, num_patches=args.num_patches, d_model=128)

    if os.path.exists(args.mamba_weights):
        print(f"Loading Mamba weights from {args.mamba_weights}")
        mamba.load_state_dict(torch.load(args.mamba_weights, map_location="cpu"))
    else:
        print(f"WARNING: Mamba weights not found at {args.mamba_weights}. Using random init for testing.")
        
    if os.path.exists(args.safenet_weights):
        print(f"Loading SafeNetSSM weights from {args.safenet_weights}")
        safenet.load_state_dict(torch.load(args.safenet_weights, map_location="cpu"))
    else:
        print(f"WARNING: SafeNetSSM weights not found at {args.safenet_weights}. Using random init for testing.")

    # Instantiate the wrapper and move to device
    print("Building SeiSM (Frozen base + Trainable MLP)...")
    fused_model = SeiSM(mamba_model=mamba, safenet_model=safenet, mlp_hidden_dim=32, final_classes=4).to(device)

    # ==========================================
    # 2. Setup Optimizer and Loss
    # ==========================================
    # CRITICAL: Only pass the MLP parameters to the optimizer!
    optimizer = optim.Adam(fused_model.mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # 3. Data Setup (TODO: Replace with actual dataloader)
    # ==========================================
    # For now, we generate dummy multimodal data that matches the shapes SafeNetSSM and Mamba2 expect.
    num_samples = 128
    
    class DummyMultimodalDataset(torch.utils.data.Dataset):
        def __len__(self): return num_samples
        def __getitem__(self, idx):
            waveforms = torch.randn(3, 8192) # Mamba input
            safe_inputs = {
                "catalog": torch.randn(10, 86, 282), # SafeNet catalog history
                "maps": torch.randn(10, args.num_patches, 50, 50, 5) # SafeNet maps history
            }
            targets = torch.randint(0, 4, (args.num_patches,)) # Labels per patch
            return waveforms, safe_inputs, targets

    dataset = DummyMultimodalDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_val_f1 = -1.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        fused_model.train() # The overridden train() keeps base models in eval()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training MLP")
        for waves, safe_in, targets in train_bar:
            waves = waves.to(device)
            safe_in = {k: v.to(device) for k, v in safe_in.items()}
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: outputs (Batch, Patches, 4)
            outputs = fused_model(waves, safe_in)
            
            # Reshape for CrossEntropyLoss
            outputs_flat = outputs.view(-1, 4)
            targets_flat = targets.view(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_fusion(fused_model, val_loader, criterion, device, desc="Validation")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            # We ONLY save the MLP weights to save space, since base models are frozen
            torch.save(fused_model.mlp.state_dict(), args.save_path)
            print(f"*** New best MLP weights saved to {args.save_path} (F1: {best_val_f1:.4f}) ***")

    print("\nTraining complete. Only the MLP head was updated.")
    
if __name__ == "__main__":
    main()
