import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class CustomAudioDataset(Dataset):
    def __init__(self, samples_dir, target_sr=32000, duration=3.0):
        self.samples_dir = samples_dir
        self.target_sr = target_sr
        self.duration = duration
        self.target_length = int(target_sr * duration)
        
        self.classes = sorted([d.name for d in os.scandir(samples_dir) if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.files = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(samples_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith(('.mp3', '.wav', '.flac')):
                    self.files.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])
                    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio
        # res_type removed as resampy is missing
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        
        # Pad or crop to target length
        if len(y) > self.target_length:
            # Random crop
            max_offset = len(y) - self.target_length
            offset = np.random.randint(0, max_offset)
            y = y[offset:offset+self.target_length]
        else:
            # Pad with zeros (silence) or repeating the sound could also work
            # For short transient sounds, just pad with silence at the end
            pad_len = self.target_length - len(y)
            y = np.pad(y, (0, pad_len), 'constant')
            
        # Data Augmentation: Random Gain
        gain = np.random.uniform(0.5, 1.5)
        y = y * gain
        
        # Normalize slightly to prevent clipping and keep range consistent
        max_val = np.max(np.abs(y))
        if max_val > 1.0:
            y = y / max_val
            
        return torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloader(samples_dir, batch_size=16, num_workers=0):
    dataset = CustomAudioDataset(samples_dir)
    
    # Calculate weights for Balanced Sampling (due to large background dataset)
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = [class_weights[label] for label in dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights), # or we could artificially increase epoch size
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers,
        drop_last=True # PANNs sometimes dislikes partial batches depending on setup
    )
    
    return dataloader, dataset.classes
