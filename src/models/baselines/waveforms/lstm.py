import torch 
import torch.nn as nn

class BiWaveformLSTM(nn.Module):
    def __init__(self, in_channels=3, d_model=128, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(BiWaveformLSTM, self).__init__()
        
        # (Batch, 3, 8192) -> (Batch, d_model, 1024)
        self.cnn_embedder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16, stride=4, padding=6),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=16, stride=2, padding=7),
            nn.GELU()
        )
        
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x: (Batch, 3, 8192)
        x = self.cnn_embedder(x)             # (Batch, d_model, 1024)
        x = x.transpose(1, 2).contiguous()   # (Batch, 1024, d_model)  
        
        # lstm_out shape: (Batch, 1024, hidden_size * 2)
        lstm_out, _ = self.lstm(x)           
        
        pooled = lstm_out.mean(dim=1)        # (Batch, hidden_size * 2)
        
        magnitude = self.regressor(pooled)   # (Batch, 1)
        
        return magnitude