# src/main.py

from models.srcnn import SRCNN
from utils.image_utils import load_image, save_image

def main():
    # Load input image
    input_image_path = "data/input_images/input.jpg"
    output_image_path = "data/output_images/output.jpg"
    input_image = load_image(input_image_path)

    # Initialize SRCNN model
    srcnn_model = SRCNN(num_channels=input_image.shape[2])

    # Perform image enhancement
    enhanced_image = enhance_image(srcnn_model, input_image)

    # Save enhanced image
    save_image(enhanced_image, output_image_path)

    print("Image enhancement complete. Enhanced image saved to", output_image_path)

def enhance_image(model, image):
    """
    Enhance image using the SRCNN model.
    
    Args:
        model (SRCNN): Trained SRCNN model.
        image (numpy.ndarray): Input image as a NumPy array.
    
    Returns:
        enhanced_image (numpy.ndarray): Enhanced image as a NumPy array.
    """
    # Convert image to tensor and add batch dimension
    image_tensor = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)

    # Apply model to input image
    with torch.no_grad():
        enhanced_image_tensor = model(image_tensor)

    # Convert enhanced image tensor back to NumPy array
    enhanced_image = enhanced_image_tensor.squeeze(0).permute(1, 2, 0).numpy().astype('uint8')

    return enhanced_image

if __name__ == "__main__":
    main()
