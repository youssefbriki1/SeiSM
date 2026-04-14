import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class WaveformTransformer(nn.Module):
    def __init__(self, in_channels=3, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2, output_size=1):
        super(WaveformTransformer, self).__init__()
        
        # (Batch, 3, 8192) -> (Batch, d_model, 1024)
        self.cnn_embedder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16, stride=4, padding=6),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=16, stride=2, padding=7),
            nn.GELU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=1024)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x: (Batch, 3, 8192)
        x = self.cnn_embedder(x)             # (Batch, d_model, 1024)
        x = x.transpose(1, 2).contiguous()   # (Batch, 1024, d_model)
        
        x = self.pos_encoder(x)              # (Batch, 1024, d_model)
        
        # Pass through the transformer encoder
        transformer_out = self.transformer_encoder(x)  # (Batch, 1024, d_model)
        
        # Pool over the sequence dimension, similar to QuakeWaveMamba2 and LSTM
        pooled = transformer_out.mean(dim=1) # (Batch, d_model)
        
        magnitude = self.regressor(pooled)   # (Batch, 1)
        return magnitude
