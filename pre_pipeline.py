import cv2
import numpy as np
import os
from tqdm import tqdm

def resize_with_padding(image, target_size=(512, 512)):
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate the scaling factor to resize the image while maintaining aspect ratio
    scaling_factor = min(target_w/original_w, target_h/original_h)
    
    # Resize the image
    new_w = int(original_w * scaling_factor)
    new_h = int(original_h * scaling_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a new image of the target size and fill it with black (or any color for padding)
    padded_image = np.zeros((target_h, target_w), dtype=np.uint8)  # Changed to 1 channel for grayscale
    
    # Calculate padding to center the resized image
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    # Place the resized image in the center of the padded image
    padded_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_image
    
    return padded_image

def process_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize and pad the grayscale image
        padded_image = resize_with_padding(gray_image)
        
        # Save the processed image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, padded_image)

if __name__ == "__main__":
    input_folder = r"C:\Users\sujal\OneDrive\Desktop\meesho_hackathon\meesho_db\train_images"
    output_folder = r"C:\Users\sujal\OneDrive\Desktop\meesho_hackathon\pre_images"
    
    process_images(input_folder, output_folder)
