import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinnedWeightedMSELoss(nn.Module):
    def __init__(self, bin_edges, bin_counts, smooth_weights=True):
        super().__init__()
        self.bin_edges = bin_edges
        
        counts = np.array(bin_counts)
        counts = np.where(counts == 0, 1, counts) 
        
        if smooth_weights:
            counts = np.sqrt(counts)
            
        total_samples = sum(counts)
        num_bins = len(counts)
        
        weights = total_samples / (num_bins * counts)
        self.weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        bin_indices = torch.bucketize(target, self.bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, len(self.weights) - 1)
        
        sample_weights = self.weights.to(target.device)[bin_indices]
        
        weighted_loss = mse_loss * sample_weights
        return weighted_loss.mean()

