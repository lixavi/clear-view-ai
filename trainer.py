import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001, device="cuda"):
    """
    Train SRCNN model.
    
    Args:
        model (nn.Module): SRCNN model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs for training (default: 10).
        learning_rate (float): Learning rate for optimizer (default: 0.001).
        device (str): Device to run training on (default: "cuda" if available, else "cpu").
    """
    # Move model to device
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Iterate over training batches
        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # Print average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
