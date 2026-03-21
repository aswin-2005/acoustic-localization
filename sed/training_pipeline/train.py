import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import *
from sed.model.main import Cnn14
from sed.training_pipeline.dataset import get_dataloader

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    batch_size = 16
    epochs = 15
    learning_rate = 1e-4

    print("Loading dataset...")
    dataloader, classes = get_dataloader(SAMPLES_DIR, batch_size=batch_size, num_workers=0)
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    # 2. Save Custom Class Labels mapping
    custom_labels_df = pd.DataFrame({
        'index': range(num_classes),
        'mid': [f"/m/custom_{i}" for i in range(num_classes)],
        'display_name': classes
    })
    custom_labels_csv = os.path.join(ROOT_DIR, "sed", "model", "labels", "custom_class_labels_indices.csv")
    custom_labels_df.to_csv(custom_labels_csv, index=False)
    print(f"Saved custom labels mapping to {custom_labels_csv}")

    # 3. Load Base Model
    print("Initializing base PANN model...")
    base_model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527  # Original AudioSet classes
    )
    checkpoint = torch.load(SED_MODEL_WEIGHTS, map_location=device)
    base_model.load_state_dict(checkpoint['model'])

    # 4. Modify Model for Custom Classes
    print(f"Modifying final layer for {num_classes} custom classes...")
    # Replace the final fully connected layer
    in_features = base_model.fc_audioset.in_features
    base_model.fc_audioset = nn.Linear(in_features, num_classes, bias=True)
    base_model.classes_num = num_classes

    # Move to device
    model = base_model.to(device)

    # Freeze convolutional layers (optional but recommended for small datasets)
    # Let's freeze all except the last couple of conv blocks and the FC layers
    for name, param in model.named_parameters():
        if "conv_block5" in name or "conv_block6" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 5. Training Setup
    criterion = nn.BCELoss()  # PANNs outputs sigmoid probabilities
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 6. Training Loop
    print("\nStarting Training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (audio_batch, label_batch) in enumerate(dataloader):
            audio_batch = audio_batch.to(device)
            label_batch = label_batch.to(device)

            # One-hot encode targets for BCE Loss
            targets_one_hot = torch.zeros(batch_size, num_classes).to(device)
            targets_one_hot.scatter_(1, label_batch.unsqueeze(1), 1.0)

            # Forward pass
            optimizer.zero_grad()
            output_dict = model(audio_batch)
            predictions = output_dict['clipwise_output']

            # Calculate loss
            loss = criterion(predictions, targets_one_hot)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Accuracy metric (argmax)
            _, predicted = torch.max(predictions, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"==> Epoch [{epoch+1}/{epochs}] Summary - Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(ROOT_DIR, "sed", "model", "weights", "custom_Cnn14.pth")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"    * Saved best model step to {save_path}")

    print("\nTraining Complete!")

if __name__ == "__main__":
    train_model()
