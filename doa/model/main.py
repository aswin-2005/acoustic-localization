import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLModel(nn.Module):
    """
    Simple 1D CNN for Sound Source Localization.
    Input shape: (Batch, 3, 33) 
    Output shape: (Batch, 3) -> unit vector [x, y, z]
    """
    def __init__(self, input_len=33):
        super(SSLModel, self).__init__()
        
        # 1D Conv across the lag dimension (dim 2)
        # We treat the 3 pairs as 'channels'
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3) # Output: x, y, z
        
    def forward(self, x):
        # x: (Batch, 3, 33)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1) # (Batch, 128)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # L2 Normalize the output to ensure it is a unit vector
        x = F.normalize(x, p=2, dim=1)
        
        return x

def angular_loss(y_pred, y_true):
    """
    Computes angular error between predicted and true unit vectors.
    Loss = 1 - cos(theta)
    """
    # Dot product since vectors are normalized
    dot_product = torch.sum(y_pred * y_true, dim=1)
    # Clip for numerical stability
    dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # We want to maximize dot product (minimize 1 - dot)
    return torch.mean(1 - dot_product)
