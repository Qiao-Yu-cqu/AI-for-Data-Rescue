import os
import cv2
from PIL import Image

def convert_to_grayscale(image):
    """
    Convert a PIL Image to grayscale.
    """
    grayscale_img = image.convert("L")
    return grayscale_img

def resize_image(image, size=(512, 512)):
    """
    Resize the input image to the given size using LANCZOS resampling.
    """
    resized_img = image.resize(size, Image.Resampling.LANCZOS)
    return resized_img

def process_images(input_folder, output_folder):
    """
    Process all images in the input folder: convert to grayscale, resize to 512x512, and save in output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path}...")

            # Open the image using PIL
            pil_image = Image.open(image_path)

            # Convert to grayscale
            grayscale_image = convert_to_grayscale(pil_image)

            # Resize to 512x512
            resized_image = resize_image(grayscale_image)

            # Save the processed image in the output folder
            output_path = os.path.join(output_folder, filename)
            resized_image.save(output_path)
            print(f"Saved processed image to {output_path}")
# Example usage:
input_folder = '/Users/yuqiao/Desktop/background_expanded'
output_folder = '/Users/yuqiao/Desktop/background_expanded_gray'
process_images(input_folder, output_folder)
