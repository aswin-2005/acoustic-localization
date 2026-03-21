import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from doa.model.main import SSLModel, angular_loss
from doa.training_pipeline.train_config import *
from config import DOA_DATA_DIR

class SSLDataset(Dataset):
    def __init__(self, data_dir):
        self.feat_dir = os.path.join(data_dir, "features")
        labels_path = os.path.join(data_dir, "labels.json")
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = self.labels[idx]
        feat_path = os.path.join(self.feat_dir, f"{item['id']}.npy")
        features = np.load(feat_path).astype(np.float32)
        target = np.array(item['cartesian_unit']).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(target)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loaders
    train_dir = os.path.join(DOA_DATA_DIR, "dataset", "train")
    test_dir = os.path.join(DOA_DATA_DIR, "dataset", "test")
    
    train_loader = DataLoader(SSLDataset(train_dir), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(SSLDataset(test_dir),  batch_size=BATCH_SIZE, shuffle=False)
    
    model = SSLModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARning_RATE)
    
    train_losses, test_losses = [], []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total = 0
        for feats, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = angular_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
            
        train_losses.append(total / len(train_loader))
        
        # Val
        model.eval()
        v_total = 0
        with torch.no_grad():
            for feats, targets in test_loader:
                feats, targets = feats.to(device), targets.to(device)
                v_total += angular_loss(model(feats), targets).item()
        test_losses.append(v_total / len(test_loader))
        
        print(f"Loss: {train_losses[-1]:.4f} | Val: {test_losses[-1]:.4f}")
        
    # Save
    weight_dir = os.path.join(os.path.dirname(DOA_DATA_DIR), "model", "weights")
    os.makedirs(weight_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(weight_dir, "ssl_model.pth"))
    
    # Plot
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Val')
    plt.legend()
    plt.savefig("_debug/training_stats.png")

if __name__ == "__main__":
    train()
