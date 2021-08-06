import cv2

def load_image(filepath):
    """
    Load image from filepath.
    
    Args:
        filepath (str): Path to the image file.
    
    Returns:
        image (numpy.ndarray): Loaded image as a NumPy array.
    """
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Unable to load image from {}".format(filepath))
    return image

def save_image(image, filepath):
    """
    Save image to filepath.
    
    Args:
        image (numpy.ndarray): Image to be saved as a NumPy array.
        filepath (str): Path to save the image file.
    """
    cv2.imwrite(filepath, image)
