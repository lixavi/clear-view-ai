import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def test_model(model, test_loader, device="cuda"):
    """
    Test SRCNN model.
    
    Args:
        model (nn.Module): Trained SRCNN model to be tested.
        test_loader (DataLoader): DataLoader for test data.
        device (str): Device to run testing on (default: "cuda" if available, else "cpu").
    """
    # Move model to device
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Initialize variables for calculating average PSNR
    total_psnr = 0.0
    num_batches = len(test_loader)
    
    # Testing loop
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate PSNR for each image in the batch
            for i in range(images.size(0)):
                mse = criterion(outputs[i], images[i]).item()
                psnr = 10 * torch.log10(1 / mse)
                total_psnr += psnr.item()
    
    # Calculate average PSNR
    avg_psnr = total_psnr / (len(test_loader.dataset))
    print(f"Average PSNR: {avg_psnr:.2f} dB")
