import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import QuakeWaveMamba2, SafeNetFull, SeiSM
# ==========================================
# 1. Initialize and Load Pre-trained Weights
# ==========================================
# (Assuming QuakeWaveMamba2 and SafeNetFull are defined in your environment)
mamba = QuakeWaveMamba2()
safenet = SafeNetFull() 

# TODO: Load your pre-trained weights here before passing to the wrapper
# mamba.load_state_dict(torch.load("mamba_weights.pth"))
# safenet.load_state_dict(torch.load("safenet_weights.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the wrapper and move to device
fused_model = SeiSM(mamba_model=mamba, safenet_model=safenet).to(device)

# ==========================================
# 2. Setup Optimizer and Loss
# ==========================================
# CRITICAL: Only pass the MLP parameters to the optimizer
optimizer = optim.Adam(fused_model.mlp.parameters(), lr=1e-3)

# Assuming a classification task for the 85 regions. 
criterion = nn.CrossEntropyLoss()

# ==========================================
# 3. Dummy Data Setup (Replace with your DataLoader)
# ==========================================
batch_size = 16
num_samples = 64

# Dummy inputs
dummy_waveforms = torch.randn(num_samples, 3, 8192)
dummy_safenet_inputs = torch.randn(num_samples, 10, 85, 5) 

# Dummy targets: integers from 0 to 3 (for 4 classes), shape (Batch, 85)
dummy_targets = torch.randint(0, 4, (num_samples, 85))

dataset = TensorDataset(dummy_waveforms, dummy_safenet_inputs, dummy_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 4. Training Loop
# ==========================================
epochs = 5

for epoch in range(epochs):
    fused_model.train() # The overridden train() keeps base models in eval()
    epoch_loss = 0.0
    
    for batch_idx, (waves, safe_in, targets) in enumerate(dataloader):
        # Move data to device
        waves = waves.to(device)
        safe_in = safe_in.to(device)
        targets = targets.to(device) # Shape: (Batch, 85)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # Output shape: (Batch, 85, 4)
        outputs = fused_model(waves, safe_in)
        
        # Reshape for CrossEntropyLoss
        # CrossEntropy expects predictions as (N, C) and targets as (N)
        # where N is Batch * Patches, and C is number of classes
        outputs_reshaped = outputs.view(-1, 4)       # Shape: (Batch * 85, 4)
        targets_reshaped = targets.view(-1)          # Shape: (Batch * 85)
        
        # Compute loss
        loss = criterion(outputs_reshaped, targets_reshaped)
        
        # Backward pass (only updates the MLP)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

print("Training complete. Only the MLP head was updated.")