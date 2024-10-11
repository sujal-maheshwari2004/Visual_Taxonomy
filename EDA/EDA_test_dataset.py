# Import necessary libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the test images
test_folder = "C://Users//shaks//Downloads//visual-taxonomy//test_images"  # Update this with your actual test folder path

# Get a list of all image files in the test folder
test_images = [f for f in os.listdir(test_folder) if f.endswith(('jpg', 'png', 'jpeg'))]

# Function to load an image
def load_image(image_path):
    img = Image.open(image_path)
    return img

# Function to get basic image statistics (size and pixel values)
def image_statistics(image):
    img_array = np.array(image)
    return {
        'shape': img_array.shape,
        'size': image.size,
        'min_pixel': np.min(img_array),
        'max_pixel': np.max(img_array),
        'mean_pixel': np.mean(img_array)
    }

# Get the dimensions and pixel statistics of all images in the test folder
def image_analysis(images, folder):
    image_data = []
    for image_name in images:
        img = load_image(os.path.join(folder, image_name))
        stats = image_statistics(img)
        image_data.append({
            'file_name': image_name,
            'width': stats['size'][0],
            'height': stats['size'][1],
            'min_pixel': stats['min_pixel'],
            'max_pixel': stats['max_pixel'],
            'mean_pixel': stats['mean_pixel']
        })
    return pd.DataFrame(image_data)

# Plot the distribution of image dimensions and pixel statistics
def plot_image_data(df):
    plt.figure(figsize=(14, 6))

    # Plot width and height distributions
    plt.subplot(1, 2, 1)
    sns.histplot(df['width'], color='blue', label='Width', kde=True)
    sns.histplot(df['height'], color='red', label='Height', kde=True)
    plt.legend()
    plt.title('Distribution of Image Dimensions')

    # Plot mean pixel values distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['mean_pixel'], color='green', label='Mean Pixel Value', kde=True)
    plt.legend()
    plt.title('Distribution of Mean Pixel Values')

    plt.tight_layout()
    plt.show()

# Perform EDA on the test images
def eda_on_test_images(test_images, folder):
    print(f'Total test images: {len(test_images)}')

    # Analyze image statistics and dimensions
    image_data_df = image_analysis(test_images, folder)
    
    # Display summary statistics of image data
    print(f"Image Data Summary:\n{image_data_df.describe()}")
    
    # Plot distributions of image dimensions and pixel statistics
    plot_image_data(image_data_df)

# Perform EDA on test images
eda_on_test_images(test_images, test_folder)
