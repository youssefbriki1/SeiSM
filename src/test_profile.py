import torch
from models import QuakeWaveMamba2, WaveformLSTM, WaveformTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def profile_model(model_cls, **kwargs):
    model = model_cls(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()
    
    # Dummy data
    x = torch.randn(256, 3, 8192, device=device)
    y = torch.randn(256, 1, device=device)
    
    torch.cuda.reset_peak_memory_stats()
    
    # Forward
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)
        
    # Backward
    loss.backward()
    optimizer.step()
    
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"{model_cls.__name__} Max Mem: {mem_mb:.2f} MB")
    
    del model, optimizer, criterion, x, y, out, loss
    torch.cuda.empty_cache()

profile_model(QuakeWaveMamba2, in_channels=3, d_model=128, d_state=64, n_layers=4, headdim=32)
profile_model(WaveformTransformer, in_channels=3, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2, output_size=1)
profile_model(WaveformLSTM, in_channels=3, d_model=128, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)
