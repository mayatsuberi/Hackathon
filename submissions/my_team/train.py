from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ModelArchitecture

# --- Configuration & Hyperparameters ---
DATA_ROOT = Path("../../dataset")
OUTPUT_WEIGHTS = Path("weights.joblib")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def get_train_dataloader(data_dir: Path, image_size: int, batch_size: int) -> DataLoader:
    """
    Creates and returns the training dataloader with the robust transform pipeline.
    """
    transform_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = ImageFolder(root=str(data_dir), transform=transform_pipeline)
    print(f"Loaded {len(train_dataset)} training images.")
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """
    Handles a single epoch of training.
    Returns the average loss and accuracy for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct_preds / total_preds
    
    return epoch_loss, epoch_acc


def main():
    """
    Main orchestration function for the training pipeline.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting training pipeline on device: {device}")

    # 2. Data Preparation
    dataset_path = DATA_ROOT / "train_set"
    train_loader = get_train_dataloader(dataset_path, IMAGE_SIZE, BATCH_SIZE)

    # 3. Model, Loss, and Optimizer Initialization
    model = ModelArchitecture().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 5. Save Artifacts
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, OUTPUT_WEIGHTS)
    print(f"Saved trained weights to {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()