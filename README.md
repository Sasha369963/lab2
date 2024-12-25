import cv2
import numpy as np
import os

# Create output directory
os.makedirs("results", exist_ok=True)

# Function for Niblack Thresholding
def niblack_threshold(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)
    return (image > mean + k * std).astype(np.uint8) * 255

# Function for Sauvola Thresholding
def sauvola_threshold(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)
    R = std.max()
    return (image > mean * (1 + k * (std / R - 1))).astype(np.uint8) * 255

# Function for Christian Thresholding
def christian_threshold(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)
    R = std.max()
    return (image > mean - k * (std / R)).astype(np.uint8) * 255

# Load images
images = ["text1.jpg", "text2.jpg", "object1.jpg", "object2.jpg"]
window_size = 128
k_values = [0.2, 0.5]

for img_name in images:
    # Read and convert to grayscale
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading {img_name}. Ensure the file exists.")
        continue

    # Global Thresholding (Otsu)
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"results/{img_name.split('.')[0]}_otsu.jpg", otsu_thresh)

    # Adaptive Thresholding (Niblack)
    niblack_thresh = niblack_threshold(img, window_size, k=0.2)
    cv2.imwrite(f"results/{img_name.split('.')[0]}_niblack.jpg", niblack_thresh)

    # Adaptive Thresholding (Sauvola)
    sauvola_thresh = sauvola_threshold(img, window_size, k=0.5)
    cv2.imwrite(f"results/{img_name.split('.')[0]}_sauvola.jpg", sauvola_thresh)

    # Adaptive Thresholding (Christian)
    christian_thresh = christian_threshold(img, window_size, k=0.5)
    cv2.imwrite(f"results/{img_name.split('.')[0]}_christian.jpg", christian_thresh)

print("Processing complete. Results are saved in the 'results' folder.")

# Save requirements.txt
with open("requirements.txt", "w") as f:
    f.write("opencv-python\n")
    f.write("numpy\n")