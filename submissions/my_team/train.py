import sys
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from augmentations import get_robust_transforms

# --- Dynamic Path Resolution ---
# Ensures that the project root is in the system path so we can import local modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ModelArchitecture
from base_model import ImageNetSubset

# --- Configuration & Hyperparameters ---
DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_WEIGHTS = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_DIM = 60 

# ImageNet standard normalization values required by the evaluator
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_dataloader(data_root: Path, image_size: int, batch_size: int) -> DataLoader:
    """
    Creates and returns the training dataloader.
    Uses the robust transform pipeline from augmentations.py to prevent 
    overfitting to backgrounds and to match the evaluation crop size.
    """
    transform_pipeline = get_robust_transforms(image_size)
    
    train_dataset = ImageNetSubset(data_root, split="training_set", transform=transform_pipeline)
    print(f"Loaded {len(train_dataset)} training images.")
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """
    Handles a single epoch of training.
    Iterates over the dataloader, performs forward and backward passes,
    updates model weights, and calculates epoch statistics.
    
    Returns:
        A tuple containing the average loss and accuracy for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in dataloader:
        # Move data to the active device (GPU/MPS/CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate batch statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
    # Calculate epoch averages
    epoch_loss = running_loss / total_preds
    epoch_acc = 100 * correct_preds / total_preds
    
    return epoch_loss, epoch_acc


def main():
    """
    Main orchestration function for the training pipeline.
    Initializes the model, data loaders, and runs the training loop.
    Finally, saves the model weights to the required joblib format.
    """
    # 1. Setup Device (Supports NVIDIA CUDA, Apple MPS, and standard CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting training pipeline on device: {device}")

    # 2. Data Preparation
    train_loader = get_train_dataloader(DATA_ROOT, IMAGE_SIZE, BATCH_SIZE)

    # 3. Model, Loss, and Optimizer Initialization
    # Initializing the model with the dynamic hidden_dim parameter required by main
    model = ModelArchitecture(num_classes=20, hidden_dim=HIDDEN_DIM, image_size=IMAGE_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 5. Save Artifacts
    # Move model back to CPU before saving to ensure compatibility with the automated grader
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, OUTPUT_WEIGHTS)
    print(f"Saved trained weights to {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()