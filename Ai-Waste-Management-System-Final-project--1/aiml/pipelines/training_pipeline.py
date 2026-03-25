import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import sys
from pathlib import Path

# Fix python path if running as a standalone script
sys.path.append(str(Path(__file__).parent.parent.parent))

from aiml.utils.logger import get_logger
from aiml.utils.preprocessing import get_image_transforms
from aiml.models.waste_classifier.model import WasteClassifier

logger = get_logger(__name__)

def train_model(data_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    """
    Trains the MobileNetV2 waste classification model.
    Expects data_dir to contain 'train' and 'val' subfolders, 
    each with subfolders for classes (wet, dry, plastic, hazardous).
    """
    logger.info(f"Starting training pipeline on {data_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Prepare datasets
    transform = get_image_transforms()
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logger.error(f"Data directories missing. Expected {train_dir} and {val_dir}")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    logger.info(f"Classes found: {train_dataset.classes}")
    
    # Initialize Model, Loss, Optimizer
    model = WasteClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            base_dir = Path(__file__).parent.parent
            save_path = base_dir / "models" / "waste_classifier" / "mobilenetv2_waste.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path} with Acc: {best_acc:.4f}")
            
    logger.info("Training complete.")

if __name__ == "__main__":
    # Example usage:
    # python training_pipeline.py
    base_dir = Path(__file__).parent.parent
    train_model(data_dir=str(base_dir / "data" / "processed"), epochs=10)
